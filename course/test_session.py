"""Service lifecycle and checkpoint cache regression tests, without dataset uploads."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch, Mock
from prediction_worker import PredictionWorker, ROOT


class WorkerLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.worker=PredictionWorker()
        self.started=[]
        def start(env):
            source='''import sys,json,os
for line in sys.stdin:
    request=json.loads(line)
    if request.get("exit"): sys.exit(0)
    result={"ok":False,"error":"test error"} if request.get("fail") else {"ok":True,"pid":os.getpid()}
    print("INSPECTION_RESULT:"+json.dumps(result),flush=True)
'''
            self.worker.process=subprocess.Popen([sys.executable,'-u','-c',source],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,text=True,encoding='utf-8',creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            self.started.append(self.worker.process)
        self.worker._start=start

    def tearDown(self):
        self.worker.close()
        self.assertTrue(all(p.poll() is not None for p in self.started))

    def test_reuse_then_explicit_release_and_restart(self):
        a=self.worker.predict({},io.StringIO())
        b=self.worker.predict({},io.StringIO())
        self.assertEqual(a['pid'],b['pid'])
        self.worker.close()
        self.assertIsNotNone(self.started[0].poll())
        self.worker.predict({},io.StringIO())
        self.assertEqual(len(self.started),2)

    def test_prediction_error_releases_worker(self):
        with self.assertRaisesRegex(RuntimeError,'test error'):
            self.worker.predict({'fail':True},io.StringIO())
        self.assertIsNone(self.worker.process)
        self.assertTrue(self.worker.predict({},io.StringIO())['ok'])

    def test_unexpected_eof_retries_once_then_reports_failure(self):
        with self.assertRaisesRegex(RuntimeError,'exited'):
            self.worker.predict({'exit':True},io.StringIO())
        self.assertEqual(len(self.started),2)
        self.assertIsNone(self.worker.process)

    def test_expired_process_is_restarted(self):
        self.worker.predict({},io.StringIO())
        self.worker.process.stdin.write('{"exit":true}\n')
        self.worker.process.stdin.flush()
        self.worker.process.wait(timeout=10)
        self.assertTrue(self.worker.predict({},io.StringIO())['ok'])
        self.assertEqual(len(self.started),2)


class CacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from predict import PredictionSession
        cls.Session=PredictionSession

    def test_checkpoint_metadata_device_switch_invalidation_and_one_slot(self):
        import predict
        with tempfile.TemporaryDirectory(dir=ROOT/'course/outputs') as folder:
            root=Path(folder)
            for name in ('a','b'):
                run=root/'course/outputs'/name
                run.mkdir(parents=True)
                (run/'baseline.ckpt').write_bytes(b'fake')
                (run/'metrics.json').write_text('{"method":"baseline"}')
            session=self.Session()
            with patch.object(predict,'ROOT',root), patch.object(predict.Patchcore,'load_from_checkpoint',side_effect=lambda *a,**k:object()) as load, patch.object(predict.torch.cuda,'is_available',return_value=True):
                a,_,hit=session.load('a','cpu');self.assertFalse(hit)
                same,_,hit=session.load('a','cpu');self.assertTrue(hit);self.assertIs(same,a)
                b,_,hit=session.load('b','cpu');self.assertFalse(hit);self.assertIsNot(b,a)
                (root/'course/outputs/b/baseline.ckpt').write_bytes(b'changed checkpoint')
                changed,_,hit=session.load('b','cpu');self.assertFalse(hit);self.assertIsNot(changed,b)
                (root/'course/outputs/b/metrics.json').write_text('{"method":"baseline", "changed":true}')
                _,_,hit=session.load('b','cpu');self.assertFalse(hit)
                _,_,hit=session.load('b','cuda');self.assertFalse(hit)
                self.assertEqual(load.call_count,5)
            session.clear();self.assertIsNone(session.model)

    def test_run_path_escape_rejected_before_loading(self):
        for run in ('../outside',str(ROOT),'..'):
            with self.assertRaises(ValueError):
                self.Session().load(run,'cpu')


class GuiSchedulingTests(unittest.TestCase):
    def test_prediction_is_serialized_and_training_releases_worker(self):
        import gui
        import threading
        with tempfile.TemporaryDirectory(dir=ROOT/'course/outputs') as folder:
            output=Path(folder)
            run=output/'run';run.mkdir()
            (run/'metrics.json').write_text('{}')
            (run/'baseline.ckpt').write_bytes(b'unit-test-placeholder')
            (output/'uploads').mkdir()
            (output/'uploads/test.png').write_bytes(b'not-an-image-worker-is-mocked')
            entered,release=threading.Event(),threading.Event()
            def predict(*args):
                entered.set()
                self.assertTrue(release.wait(timeout=5))
                return {'timing':{'cache_hit':True}}
            worker=Mock();worker.predict.side_effect=predict
            process=Mock();process.poll.return_value=None
            with patch.object(gui,'OUTPUT',output), patch.object(gui,'JOB',{}), patch.object(gui,'PROCESS',None), patch.object(gui,'PREDICTION_THREAD',None), patch.object(gui,'PREDICTOR',worker), patch.object(gui,'datasets',return_value=[{'id':'bottle','ready':True}]), patch.object(gui.subprocess,'Popen',return_value=process) as spawn:
                try:
                    job=gui.launch({'task':'predict','run':'run','image':'test.png','device':'cpu'})
                    self.assertEqual(job['status'],'running')
                    self.assertTrue(entered.wait(timeout=3))
                    with self.assertRaisesRegex(ValueError,'已有任务'):
                        gui.launch({'task':'train','dataset':'bottle','method':'baseline'})
                finally:
                    release.set()
                    gui.PREDICTION_THREAD.join(timeout=5)
                self.assertEqual(gui.job_state()['status'],'completed')
                self.assertTrue(gui.job_state()['cache_hit'])
                def start(*args,**kwargs):
                    worker.close.assert_called_once()
                    return process
                spawn.side_effect=start
                gui.launch({'task':'train','dataset':'bottle','method':'baseline'})
                self.assertEqual(gui.JOB['status'],'running')
                spawn.assert_called_once()

    def test_worker_failure_sets_failed_job(self):
        import gui
        with tempfile.TemporaryDirectory(dir=ROOT/'course/outputs') as folder:
            output=Path(folder);run=output/'run';run.mkdir()
            (run/'metrics.json').write_text('{}');(run/'baseline.ckpt').write_bytes(b'fake')
            (output/'uploads').mkdir();(output/'uploads/test.png').write_bytes(b'fake')
            worker=Mock();worker.predict.side_effect=RuntimeError('worker test failure')
            with patch.object(gui,'OUTPUT',output), patch.object(gui,'JOB',{}), patch.object(gui,'PROCESS',None), patch.object(gui,'PREDICTION_THREAD',None), patch.object(gui,'PREDICTOR',worker):
                gui.launch({'task':'predict','run':'run','image':'test.png','device':'cpu'})
                gui.PREDICTION_THREAD.join(timeout=5)
                state=gui.job_state()
                self.assertEqual(state['status'],'failed')
                self.assertIn('worker test failure',state['log'])


if __name__=='__main__':
    unittest.main()
