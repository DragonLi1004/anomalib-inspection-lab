const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = value => value == null ? '—' : Number(value).toFixed(4);
const size = value => value == null ? '—' : value < 1048576 ? (value / 1024).toFixed(1) + ' KB' : (value / 1048576).toFixed(2) + ' MB';
const imageURL = path => '/api/file?path=' + encodeURIComponent(path);
let state, currentImages = [], imageIndex = 0, selectedRun = '', uploadID = '', pollBusy = false, previousJob = '';
function toast(message) { $('toast').textContent = message; $('toast').classList.remove('hidden'); clearTimeout(toast.timer); toast.timer = setTimeout(()=>$('toast').classList.add('hidden'), 6000); }
async function api(path, body) { const r = await fetch(path, body ? {method:'POST',headers:{'Content-Type':'application/json','X-Lab-Token':state.token},body:JSON.stringify(body)} : {}); const data = await r.json(); if(!r.ok) throw new Error(data.error || '请求失败'); return data; }
function page(name) { document.querySelectorAll('.page').forEach(e=>e.classList.toggle('active',e.id===name)); document.querySelectorAll('.nav').forEach(e=>e.classList.toggle('active',e.dataset.page===name)); $('page-title').textContent = {overview:'实验总览',detect:'图像检测',experiments:'对比实验',storage:'无损部署'}[name]; location.hash=name; }
document.querySelectorAll('[data-page]').forEach(e=>e.onclick=()=>page(e.dataset.page));
document.querySelectorAll('[data-goto]').forEach(e=>e.onclick=()=>page(e.dataset.goto));
function options(select, items, label) { const old=select.value; const html=items.map(item=>`<option value="${esc(item.id)}" ${item.ready===false?'disabled':''}>${esc(label(item))}</option>`).join(''); if(select.innerHTML!==html) { select.innerHTML=html; if(items.some(i=>i.id===old)) select.value=old; } }
function runLabel(run) { return run.display_name || run.id; }
function datasetLabel(id) { return state.catalog.datasets[id]?.label || id; }
function methodLabel(id) { const m=state.catalog.methods[id]; return m?`${m.name} · ${m.label}`:id; }
function namedRun(id) { const run=state.runs.find(r=>r.id===id); return run?runLabel(run):id; }
function renderPredictModels() {
 const dataset=$('predict-dataset').value;
 const rows=state.runs.filter(r=>r.dataset===dataset).sort((a,b)=>{
  const priority=r=>(r.protocol_version==='paired-coreset-v2'?0:2)+(r.method==='baseline'?0:1);
  return priority(a)-priority(b);
 });
 options($('predict-run'),rows,runLabel);
 const selected=rows.find(r=>r.id===$('predict-run').value);
 const info=state.catalog.datasets[dataset];
 $('model-scope').textContent=info?`${info.scope} · 样本包含：${info.examples}。输出异常分数、异常区域，不分类缺陷名称。${selected?' 当前：'+selected.method_status+'；'+selected.protocol_label+'。':''}`:'该类别暂无本地模型';
}
$('predict-dataset').onchange=()=>{ $('predict-run').innerHTML=''; renderPredictModels(); };
$('predict-run').onchange=renderPredictModels;
function showImage() { const path=currentImages[imageIndex]; $('hero-image').classList.toggle('empty',!path); $('hero-image').innerHTML=path?`<img src="${imageURL(path)}" alt="原图、真实缺陷标注、异常热力图和预测缺陷区域">`:'该实验暂未生成图片'; $('image-caption').textContent=path?`${imageIndex+1} / ${currentImages.length} · ${path.split('/').slice(-2).join('/')}`:'—'; $('prev-image').disabled=$('next-image').disabled=!path; }
async function loadRun() { selectedRun=$('preview-run').value; const run=state.runs.find(r=>r.id===selectedRun); $('current-auroc').textContent=run?fmt(run.metrics[0].image_AUROC):'—'; $('current-label').textContent=run?`${datasetLabel(run.dataset)} · ${run.model_name} · 测试 ${run.split_counts.test} 张`:'暂无实验'; $('bank-size').textContent=run?size(run.memory_bank_bytes):'—'; if(!run)return; try {const data=await api('/api/run?id='+encodeURIComponent(selectedRun)); currentImages=data.images; imageIndex=0;showImage();}catch(e){toast(e.message);} }
$('preview-run').onchange=loadRun;
$('prev-image').onclick=()=>{imageIndex=(imageIndex-1+currentImages.length)%currentImages.length;showImage();};
$('next-image').onclick=()=>{imageIndex=(imageIndex+1)%currentImages.length;showImage();};
document.body.addEventListener('click',event=>{if(event.target.matches('.hero-image img,.prediction-results img')){$('image-dialog').querySelector('img').src=event.target.src;$('image-dialog').showModal();}});
$('close-dialog').onclick=()=>$('image-dialog').close();
function comparison() {
 const opt=state.runs.find(r=>r.id===$('compare-run').value);
 if(!opt){$('comparison').innerHTML='完成原始方法与 PCA 实验后，显示对比结果';return;}
 const keys=['dataset','seed','device','backbone','coreset_sampling_ratio','num_neighbors','val_split_ratio','split_sha256','protocol_version','coreset_sha256','course_code_sha256'];
 const base=state.runs.find(r=>r.method==='baseline'&&opt.protocol_version==='paired-coreset-v2'&&opt.split_sha256&&keys.every(k=>r[k]===opt[k])&&JSON.stringify(r.image_size)===JSON.stringify(opt.image_size));
 if(!base){$('comparison').innerHTML='尚无经相同特征库校验的新版本基线。旧记录保留用于查看，请重新运行同配置 baseline 与 PCA。';return;}
 const a=base.metrics[0],b=opt.metrics[0];
 function bars(label,x,y,text,unit,lower=false){const max=Math.max(x,y,1e-9);const delta=y-x;return `<div class="compare-row"><div class="compare-row-title"><span>${label}</span><span class="delta ${(lower?delta>0:delta<0)?'bad':''}">${text}</span></div><div class="compare-note">原始 ${unit(x)} <span>→</span> ${esc(methodLabel(opt.method))} ${unit(y)}</div><div class="compare-bar"><i style="width:${x/max*100}%"></i></div><div class="compare-bar"><i class="optimized" style="width:${y/max*100}%"></i></div></div>`;}
 let html=bars('检索状态占用（特征库 + 投影）',base.search_state_bytes,opt.search_state_bytes,`${Math.abs((1-opt.search_state_bytes/base.search_state_bytes)*100).toFixed(1)}% ${opt.search_state_bytes<=base.search_state_bytes?'减少':'增加'}`,size,true);
 html+=bars('图像 AUROC',a.image_AUROC,b.image_AUROC,`${((b.image_AUROC-a.image_AUROC)*100).toFixed(2)} 个百分点`,fmt);
 html+=bars('像素 AUROC',a.pixel_AUROC,b.pixel_AUROC,`${((b.pixel_AUROC-a.pixel_AUROC)*100).toFixed(2)} 个百分点`,fmt);
 html+=`<p class="compare-note">图像 F1：${fmt(a.image_F1Score)} → ${fmt(b.image_F1Score)}<br>单图模型前向中位耗时：${base.latency_median_ms.toFixed(2)} → ${opt.latency_median_ms.toFixed(2)} ms（不含读图、预处理和显示；单次运行会受负载影响）。<br>投影额外占用：${size(opt.projection_bytes)}。模型文件：${size(base.checkpoint_bytes)} → ${size(opt.checkpoint_bytes)}。<br>种子 ${opt.seed} · 训练 ${opt.split_counts.train} / 验证 ${opt.split_counts.val} / 测试 ${opt.split_counts.test}。<br>当前仅展示这组实验，不能推断所有类别或工业场景的效果。</p>`;
 $('comparison').classList.remove('empty');$('comparison').innerHTML=html;
}
$('compare-run').onchange=comparison;
function render() {
 $('run-count').textContent=state.runs.length;
 $('image-count').textContent=state.datasets.reduce((a,d)=>a+d.images,0).toLocaleString();
 options($('preview-run'),state.runs,runLabel);
 options($('predict-dataset'),Object.entries(state.catalog.datasets).filter(([id])=>state.runs.some(r=>r.dataset===id)).map(([id,d])=>({id,...d})),d=>`${d.label} · ${d.scope}`);
 renderPredictModels();
 options($('train-method'),Object.entries(state.catalog.methods).map(([id,m])=>({id,...m})),m=>`${m.name} · ${m.status}`);
 $('optimization-catalog').innerHTML=state.catalog.optimizations.map(o=>`<div><h3>${esc(o.name)}</h3><p>${esc(o.scope)} · ${esc(o.status)}</p><p>${esc(o.benefit)}</p></div>`).join('');
 options($('compare-run'),state.runs.filter(r=>r.method!=='baseline'),runLabel);
 options($('train-dataset'),state.datasets,d=>`${d.label} · ${d.images} 张${d.ready?'':'（下载中 / 未下载）'}`);
 $('datasets').innerHTML=state.datasets.map(d=>`<div class="dataset-row"><div class="dataset-top"><span>${esc(d.label)}</span><small>${d.images} / ${d.expected} · ${d.ready?'可用':'待补齐'}</small></div><div class="progress"><i style="width:${Math.min(100,d.images/d.expected*100)}%"></i></div></div>`).join('');
 $('runs-table').innerHTML=state.runs.slice(0,12).map(r=>`<tr><td>${esc(datasetLabel(r.dataset))}<small>${esc(r.id.slice(-15))} · seed ${r.seed}</small></td><td><span class="method-chip">${esc(methodLabel(r.method))}</span></td><td>${r.split_counts.train} / ${r.split_counts.val} / ${r.split_counts.test}</td><td>${fmt(r.metrics[0].image_AUROC)}</td><td>${fmt(r.metrics[0].image_F1Score)}</td><td>${fmt(r.metrics[0].pixel_AUROC)}</td><td>${size(r.memory_bank_bytes)}</td></tr>`).join('')||'<tr><td colspan="7">暂无实验，先运行原始基线。</td></tr>';
 if(selectedRun!==$('preview-run').value)loadRun();
 comparison();
 if(state.predictions.length){$('prediction-results').classList.remove('empty');$('prediction-results').innerHTML=state.predictions.slice(0,3).map(p=>`<div class="prediction-card">${p.records.map(r=>`<span class="result-badge ${r.anomalous?'alert':''}">${r.anomalous?'检测到异常':'未达到异常阈值'}</span> · 得分 ${Number(r.anomaly_score).toFixed(4)}`).join('')}<p>${esc(p.id)}${p.records[0]?.model_run?' · '+esc(namedRun(p.records[0].model_run)):''}</p>${p.timing?.request_seconds!=null?`<p class="compare-note">${p.timing.cache_hit?'复用模型':'加载模型'} · 处理 ${p.timing.request_seconds.toFixed(2)} 秒（不含进程启动）</p>`:''}${p.images.map(path=>`<img src="${imageURL(path)}" alt="单图检测结果和缺陷热力图" loading="lazy">`).join('')}</div>`).join('');}
 const job=state.job; const running=job?.status==='running'; ['train','predict','download'].forEach(id=>$(id).disabled=running);
 if(job){$('job-panel').classList.remove('hidden');$('job-title').textContent=job.label;const names={running:'正在运行',completed:'已完成',failed:'运行失败，请查看日志'};$('job-state').textContent=names[job.status]+(job.elapsed_seconds!=null?` · ${job.elapsed_seconds.toFixed(2)} 秒 · ${job.cache_hit?'已复用模型':'首次加载 / 切换模型'}`:'');$('job-log').textContent=job.log;if(previousJob==='running'&&!running){toast(names[job.status]);if(job.task==='predict')page('detect');}previousJob=job.status;}
}
function renderSummary(summary) {
 $('summary-status').textContent=`${summary.complete?'完整矩阵已完成':'实验尚未全部完成'} · ${summary.available_runs} / 27 组记录 · ${summary.matched_groups} / 9 组完整配对`;
 const stat=s=>`${s.mean.toFixed(4)} ± ${s.std==null?'—':s.std.toFixed(4)}`;
 $('summary-table').innerHTML=summary.rows.map(r=>{
  const s=r.statistics, change=r.paired_changes.image_AUROC_delta_pp.mean;
  return `<tr><td>${esc(datasetLabel(r.dataset))}<small>${esc(methodLabel(r.method))}</small></td><td>${r.n}</td><td>${stat(s.image_AUROC)}</td><td>${stat(s.image_F1Score)}</td><td>${stat(s.pixel_AUROC)}</td><td>${r.method==='baseline'?'基线':r.paired_changes.search_state_reduction_percent.mean.toFixed(2)+'%'}</td><td class="delta ${change<0?'bad':''}">${r.method==='baseline'?'—':(change>=0?'+':'')+change.toFixed(3)+' pp'}</td></tr>`;
 }).join('')||'<tr><td colspan="7">尚无完整配对：同类别、同种子需要 baseline、pca96 和 pca192 三组结果。</td></tr>';
}
let storageReport;
function renderRestoreCommand() {
 if(!storageReport)return;
 $('restore-command').textContent=`.\\.venv\\Scripts\\python.exe course/lossless_package.py restore --package "${storageReport.package}" --run "${$('restore-run').value}" --output "course/outputs/restored-${$('restore-run').value}/baseline.ckpt"`;
}
function renderStorage(report) {
 storageReport=report;
 if(!report.complete){$('storage-status').textContent='尚未生成无损模型包，请先运行 audit_lossless.py 和 report_lossless.py。';return;}
 const b=report.bundle;
 $('storage-before').textContent=(b.original_bytes/1048576).toFixed(2)+' MiB';
 $('storage-after').textContent=(b.package_bytes/1048576).toFixed(2)+' MiB';
 $('storage-saving').textContent=b.reduction_percent.toFixed(2)+'%';
 $('storage-parity').textContent=report.runs.filter(r=>r.byte_identical&&r.state_dict_identical&&r.metrics_identical).length+' / '+report.runs.length;
 $('storage-status').textContent=`${report.test_image_evaluations} 次测试图像评估完成；9 个恢复模型的四项检测指标全部与原始记录相同。${report.package_available?'':'当前机器尚无模型包，请运行打包验证命令。'}`;
 $('download-bundle').classList.toggle('hidden',!report.package_available);$('download-bundle').href='/api/package?run=bundle';
 $('storage-table').innerHTML=report.runs.map(r=>`<tr><td>${esc(datasetLabel(r.dataset))}<small>seed ${r.seed} · ${esc(r.source_run)}</small></td><td>${(r.single_package.original_bytes/1048576).toFixed(2)}</td><td>${(r.single_package.package_bytes/1048576).toFixed(2)}</td><td class="delta">${r.single_package.reduction_percent.toFixed(2)}%</td><td>${r.metrics_identical?'全部不变':'尚未通过'}</td><td>${report.package_available?`<a class="text-link" download="${esc(r.source_run)}.alpack" href="/api/package?run=${encodeURIComponent(r.source_run)}">下载 ↓</a>`:'本地未生成'}</td></tr>`).join('');
 options($('restore-run'),report.runs.map(r=>({id:r.source_run})),r=>namedRun(r.id));
 const times=report.runs.map(r=>r.restore_seconds);
 $('storage-cost').textContent=`本机打包组合包约 ${b.pack_seconds.toFixed(2)} 秒，恢复单个模型约 ${Math.min(...times).toFixed(3)}–${Math.max(...times).toFixed(3)} 秒。此项收益是部署包大小，不是推理速度或显存。`;
 $('package-location').textContent='本地组合包：'+report.package_local_path;
 renderRestoreCommand();
}
$('restore-run').onchange=renderRestoreCommand;
async function refresh(){if(pollBusy)return;pollBusy=true;try{state=await api('/api/state');$('connection-error').classList.add('hidden');render();try{renderSummary(await api('/api/summary'));}catch(error){$('summary-status').textContent='汇总不可用：'+error.message;$('summary-table').innerHTML='';}try{renderStorage(await api('/api/storage'));}catch(error){$('storage-status').textContent='存储报告不可用：'+error.message;}}catch(e){$('connection-error').classList.remove('hidden');}finally{pollBusy=false;}}
async function submit(body){try{await api('/api/job',body);toast('任务已启动，日志显示在页面下方');await refresh();$('job-panel').scrollIntoView({behavior:'smooth',block:'nearest'});}catch(e){toast(e.message);}}
$('train').onclick=()=>submit({task:'train',dataset:$('train-dataset').value,method:$('train-method').value,seed:Number($('train-seed').value),device:$('train-device').value});
$('download').onclick=()=>submit({task:'download'});
$('predict').onclick=()=>{if(!uploadID)return toast('请先选择一张图片');if(!$('predict-run').value)return toast('请先完成一个训练实验');submit({task:'predict',run:$('predict-run').value,image:uploadID,device:$('predict-device').value});};
$('image-input').onchange=async event=>{const file=event.target.files[0];if(!file)return;uploadID='';if(file.size>10000000)return toast('图片请小于 10 MB');try{const data=await new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result);reader.onerror=reject;reader.readAsDataURL(file);});const result=await api('/api/upload',{data:data.split(',')[1]});uploadID=result.id;$('upload-preview').src=data;$('upload-preview').classList.remove('hidden');$('upload-label').classList.add('hidden');}catch(e){toast(e.message);}};
$('toggle-log').onclick=()=>{$('job-log').classList.toggle('hidden');$('toggle-log').textContent=$('job-log').classList.contains('hidden')?'展开日志':'收起日志';};
$('refresh').onclick=refresh;
if(['overview','detect','experiments','storage'].includes(location.hash.slice(1)))page(location.hash.slice(1));
async function poll(){await refresh();setTimeout(poll,state?.job?.status==='running'&&state.job.task==='predict'?500:4000);}poll();
