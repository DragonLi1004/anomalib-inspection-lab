"""Lossless archive integrity, deduplication and local download boundaries."""
import hashlib,json,random,tempfile,threading,unittest,urllib.error,urllib.request,zipfile
from pathlib import Path
from baseline import ROOT
from lossless_package import pack_checkpoints,restore_checkpoint
import gui

class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'course/outputs',prefix='package-test-')
        self.folder=Path(self.temp.name)
        self.assertTrue(self.folder.resolve().is_relative_to((ROOT/'course/outputs').resolve()))
        common=random.Random(31).randbytes(100000)
        self.models={}
        for name in ['one','two']:
            p=self.folder/(name+'.ckpt')
            with zipfile.ZipFile(p,'w') as z:
                z.writestr('archive/common',common);z.writestr('archive/unique',name.encode());z.comment=b'keep this too'
            self.models[name]=p
        self.package=self.folder/'test.alpack'
        self.stats=pack_checkpoints(self.models,self.package)
    def tearDown(self):self.temp.cleanup()

    def test_roundtrip_and_cross_model_dedup(self):
        self.assertLess(self.stats['package_bytes'],self.stats['original_bytes']*.6)
        for name,p in self.models.items():
            target=restore_checkpoint(self.package,name,self.folder/(name+'-restored.ckpt'))
            self.assertEqual(p.read_bytes(),target.read_bytes())
            self.assertEqual(target,restore_checkpoint(self.package,name,target))

    def test_restore_does_not_require_original_files(self):
        expected=self.models['one'].read_bytes()
        self.models['one'].unlink()
        target=restore_checkpoint(self.package,'one',self.folder/'standalone.ckpt')
        self.assertEqual(expected,target.read_bytes())

    def test_existing_different_target_is_preserved(self):
        target=self.folder/'existing.ckpt';target.write_bytes(b'preserve')
        with self.assertRaises(FileExistsError):restore_checkpoint(self.package,'one',target)
        self.assertEqual(target.read_bytes(),b'preserve')
        with self.assertRaises(FileExistsError):pack_checkpoints(self.models,self.package)

    def test_corrupt_blob_rejected_without_publishing_checkpoint(self):
        corrupt=self.folder/'corrupt.alpack'
        with zipfile.ZipFile(self.package) as old,zipfile.ZipFile(corrupt,'w') as new:
            manifest=json.loads(old.read('manifest.json'));chosen=manifest['models']['one']['parts'][0]
            for info in old.infolist():
                data=old.read(info.filename)
                if info.filename=='blobs/'+chosen:data=bytes([data[0]^1])+data[1:]
                new.writestr(info.filename,data)
        target=self.folder/'bad.ckpt'
        with self.assertRaisesRegex(ValueError,'Corrupt'):restore_checkpoint(corrupt,'one',target)
        self.assertFalse(target.exists());self.assertFalse(target.with_name(target.name+'.partial').exists())

class StorageApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=gui.ThreadingHTTPServer(('127.0.0.1',0),gui.Handler)
        cls.url=f'http://127.0.0.1:{cls.server.server_port}'
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
    @classmethod
    def tearDownClass(cls):cls.server.shutdown();cls.server.server_close();cls.thread.join()
    def test_report_and_package_bytes(self):
        with urllib.request.urlopen(self.url+'/api/storage') as response:r=json.load(response)
        self.assertTrue(r['complete']);self.assertTrue(r['package_available'])
        with urllib.request.urlopen(self.url+'/api/package?run=bundle') as response:data=response.read()
        self.assertEqual(len(data),r['bundle']['package_bytes'])
        self.assertEqual(hashlib.sha256(data).hexdigest(),r['bundle']['package_sha256'])
    def test_package_route_cannot_read_arbitrary_files(self):
        with urllib.request.urlopen(self.url+'/api/storage') as response:r=json.load(response)
        self.assertEqual(len(r['runs']),9)
        for name in ['../baseline.ckpt','../../.runtime/gh/hosts.yml','unknown']:
            with self.assertRaises(urllib.error.HTTPError) as e:urllib.request.urlopen(self.url+'/api/package?run='+name)
            self.assertEqual(e.exception.code,404)

if __name__=='__main__':unittest.main()
