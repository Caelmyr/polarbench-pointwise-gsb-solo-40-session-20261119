import {PANEL, button, detailTable, enc, fmt, legends, metrics, option} from './panels.js';

const el = (id) => document.getElementById(id);
const copy = structuredClone;
const MAP = el('map-image');
const OVERLAY = el('selection-canvas');
const context = OVERLAY.getContext('2d');
let config, projects = [], project = null, state = null, report = null, tab = 'map', tool = 'probe';
let target = {}, regionId = '', drag = null, viewport = null, analyzedSequence = 0, captureSequence = 0;
let viewURL, evidenceURL, toastTimer, previewTimer, mapMode = 'inspection';

function toast(text, error=false) {
  el('toast').textContent = text;
  el('toast').classList.toggle('error', error);
  el('toast').hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(()=>{el('toast').hidden=true;}, error?6000:3100);
}

async function request(path, {method='GET', data, binary=false}={}) {
  const opts = {method};
  if (data !== undefined) {
    if (data instanceof FormData) opts.body = data;
    else {
      opts.headers = {'Content-Type':'application/json'};
      opts.body = JSON.stringify(data);
    }
  }
  const response = await fetch(path, opts);
  if (!response.ok) {
    const error = await response.json().catch(()=>({error:response.statusText}));
    throw new Error(error.error || `Request failed (${response.status})`);
  }
  if (binary) return {blob:await response.blob(), headers:response.headers};
  if (response.status === 204) return null;
  return response.json();
}

function forCapture(path='', method='GET', data, binary=false) {
  if (!project) throw new Error('Open a capture before using this tool.');
  return request(`/api/projects/${project.id}${path}`, {method, data, binary});
}
function processing(path, overrides={}, binary=false) {
  return forCapture(path, 'POST', {state,...overrides}, binary);
}
const number = (key) => {
  const field = el(key);
  if (!field || field.value === '') throw new Error('Enter each required value.');
  return Number(field.value);
};
const value = (key) => el(key)?.value;
const checked = (key) => !!el(key)?.checked;
const revision = () => ({revision:project.revision});
const regionSelection = (key='measure-region') => {
  const stored = value(key) || '';
  if (stored) return {region_id: stored};
  return target.box ? {geometry:{shape:'box',box:target.box}} : {};
};
const activeBox = () => {
  if (target.box) return target.box;
  const selected = project?.regions.find(row=>row.id===regionId);
  return selected?.geometry.box || null;
};
function filename(headers, fallback) {
  const content = headers.get('content-disposition') || '';
  const unicode = content.match(/filename\*=UTF-8''([^;]+)/i);
  if (unicode) return decodeURIComponent(unicode[1]);
  const basic = content.match(/filename="?([^";]+)"?/i);
  return basic?.[1] || fallback;
}
function downloadBlob(result, fallback) {
  const anchor = document.createElement('a');
  const url = URL.createObjectURL(result.blob);
  anchor.href = url;
  anchor.download = filename(result.headers,fallback);
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
}

function setTool(next) {
  tool = next;
  for (const item of el('select-tools').querySelectorAll('[data-tool]')) item.setAttribute('aria-pressed',String(item.dataset.tool===tool));
}
function renderPanel() {
  if (project) el('inspector-panel').innerHTML = PANEL[tab]({config,state,project,projects,report,selection:target,regionId});
}
function renderRail() {
  el('capture-count').textContent = projects.length;
  el('capture-list').innerHTML = projects.map(item=>`<button type="button" class="capture-nav" data-project="${item.id}" aria-current="${item.id===project?.id}"><span class="rail-preview" aria-hidden="true">${item.type==='mosaic'?'▦':'◈'}</span><span class="nav-info"><strong title="${enc(item.name)}">${enc(item.name)}</strong><small>${item.width} × ${item.height} px</small></span></button>`).join('');
  el('demo-list').innerHTML = config.demos.map(scene=>`<button class="demo-nav" data-demo="${scene.key}" type="button" title="${enc(scene.description)}"><span class="rail-preview" aria-hidden="true">↗</span><span class="nav-info"><strong>${enc(scene.name)}</strong><small>${scene.format==='mosaic'?'RAW MOSAIC':'ANALYZER PLANES'}</small></span></button>`).join('');
}
function renderHeading() {
  el('active-screen').hidden = !project;
  el('empty-screen').hidden = !!project;
  if (!project) return;
  const capture = project.capture;
  const name = capture.type === 'mosaic' ? 'Micro-polarizer mosaic' : 'Four analyzer planes';
  el('project-name').textContent = project.name;
  el('project-meta').textContent = `${name}  ·  ${capture.width} × ${capture.height} px${capture.synthetic?'  ·  Synthetic demonstration':''}`;
}
function setTab(next) {
  tab = next;
  for (const item of el('tool-tabs').querySelectorAll('[data-tab]')) item.setAttribute('aria-selected',String(item.dataset.tab===tab));
  renderPanel();
}
async function refreshProjects() {
  projects = (await request('/api/projects')).projects;
  renderRail();
}
async function useCapture(next, readId = null) {
  const seq = ++captureSequence;
  const loaded = readId ? await request(`/api/projects/${readId}`) : next;
  if (seq !== captureSequence) return;
  project = loaded;
  state = project ? copy(project.defaults) : null;
  report = null;
  target = {};
  regionId = '';
  mapMode = 'inspection';
  el('view-caption').textContent = 'LIVE INSPECTION';
  el('output-card').hidden = true;
  el('legend-card').innerHTML = '';
  el('viewport-status').textContent = '';
  setTool('probe');
  setTab('map');
  renderHeading();
  renderRail();
  if (project) {
    history.replaceState(null,'',`#capture=${project.id}`);
    await updateMap(true);
  } else {
    history.replaceState(null,'',location.pathname);
  }
}
async function reloadRecord(preserve=true) {
  const id = project.id;
  const current = await forCapture();
  if (project?.id !== id) return;
  project = current;
  if (!preserve) state = copy(project.defaults);
  renderHeading();
  renderPanel();
  redrawSelection();
}
function changeLayer(layer) {
  const item = config.layer_options.find(x=>x.id===layer);
  if (!item) throw new Error('Select a detector map.');
  state.view.layer = layer;
  state.view.display = {low:item.minimum,high:item.maximum};
  state.view.palette = item.palette;
  state.view.quality_focus = null;
}
function displayedUrl(blob) {
  const prior = viewURL;
  viewURL = URL.createObjectURL(blob);
  MAP.src = viewURL;
  if (prior) URL.revokeObjectURL(prior);
}
function applyViewport(next) {
  viewport = next;
  el('viewport-status').textContent = viewport ? `${viewport.width} × ${viewport.height} px` : '';
  redrawSelection();
}
async function updateMap(withDiagnostics=false) {
  if (!project) return;
  const seq = ++analyzedSequence;
  const id = project.id;
  const snapshot = copy(state);
  mapMode = 'inspection';
  el('view-caption').textContent = 'LIVE INSPECTION';
  el('map-loading').hidden = false;
  try {
    const [result,key] = await Promise.all([
      forCapture('/preview','POST',{state:snapshot},true),
      forCapture('/legend','POST',{state:snapshot}),
    ]);
    if (seq !== analyzedSequence || project?.id !== id) return;
    const header = result.headers.get('x-detector-viewport')?.split(',').map(Number);
    if (header?.length === 4) applyViewport({x:header[0],y:header[1],width:header[2],height:header[3]});
    displayedUrl(result.blob);
    el('frame-label').textContent = config.layer_options.find(x=>x.id===snapshot.view.layer).label;
    el('legend-card').innerHTML = legends(key);
    if (withDiagnostics) {
      const diagnostics = await forCapture('/diagnostics','POST',{state:snapshot});
      if (seq === analyzedSequence && project?.id === id) {
        report = diagnostics;
        if (['map','quality'].includes(tab)) renderPanel();
      }
    }
  } finally {
    if (seq === analyzedSequence) el('map-loading').hidden = true;
  }
}
function markViewport() {
  const w = MAP.clientWidth;
  const h = MAP.clientHeight;
  if (w > 0 && h > 0) {
    OVERLAY.width = Math.max(1, Math.round(w * window.devicePixelRatio));
    OVERLAY.height = Math.max(1, Math.round(h * window.devicePixelRatio));
  }
  redrawSelection();
}
function canvasPosition(x, y) {
  return [(x - viewport.x) / viewport.width * OVERLAY.width, (y - viewport.y) / viewport.height * OVERLAY.height];
}
function rectangle(box, color, fill, ellipse=false) {
  if (!viewport) return;
  const [x1,y1] = canvasPosition(box.x,box.y);
  const [x2,y2] = canvasPosition(box.x+box.width,box.y+box.height);
  context.beginPath();
  if (ellipse) context.ellipse((x1+x2)/2,(y1+y2)/2,(x2-x1)/2,(y2-y1)/2,0,0,Math.PI*2);
  else context.rect(x1,y1,x2-x1,y2-y1);
  if (fill) { context.fillStyle=fill; context.fill(); }
  context.strokeStyle=color;
  context.lineWidth=2 * window.devicePixelRatio;
  context.stroke();
}
function drawLine(line,color,closed=false) {
  const points = Array.isArray(line) ? line : [line.start,line.end];
  context.beginPath();
  for (const [i,p] of points.entries()) {
    const [x,y] = canvasPosition(p.x+.5,p.y+.5);
    if (i === 0) context.moveTo(x,y);
    else context.lineTo(x,y);
  }
  if (closed) context.closePath();
  context.strokeStyle=color;
  context.lineWidth=2*window.devicePixelRatio;
  context.stroke();
}
function redrawSelection() {
  context.clearRect(0,0,OVERLAY.width,OVERLAY.height);
  if (!viewport || !project || mapMode!=='inspection') return;
  const alignedMap = !state.view.layer.startsWith('raw-');
  if (alignedMap && tab==='measure') {
    for (const item of project.regions) {
      const g = item.geometry;
      const active = item.id===regionId;
      if (g.shape==='polygon') drawLine(g.points,item.color,true);
      else rectangle(g.box,item.color,active?'rgba(157,225,207,.18)':null,g.shape==='ellipse');
    }
  }
  if (target.box) rectangle(target.box,'#b6eedf','rgba(157,225,207,.13)');
  if (target.line) drawLine(target.line,'#f1ce92');
  if (target.point) {
    const p=target.point;
    const [x,y]=canvasPosition(p.x+.5,p.y+.5);
    context.strokeStyle='#f0dfb6';
    context.lineWidth=1.25*window.devicePixelRatio;
    context.beginPath();
    context.moveTo(x-9,y); context.lineTo(x+9,y);
    context.moveTo(x,y-9); context.lineTo(x,y+9);
    context.stroke();
  }
}
function fromPointer(event) {
  if (!viewport || mapMode!=='inspection') return null;
  const rect = OVERLAY.getBoundingClientRect();
  if (!rect.width || !rect.height) return null;
  const relative = (at,from,extent,length) => Math.max(0,Math.min(length-1,Math.floor((at-from)/extent*length)));
  return {x:viewport.x+relative(event.clientX,rect.left,rect.width,viewport.width),y:viewport.y+relative(event.clientY,rect.top,rect.height,viewport.height)};
}
function buildBox(a,b) {
  return {x:Math.min(a.x,b.x),y:Math.min(a.y,b.y),width:Math.abs(b.x-a.x)+1,height:Math.abs(b.y-a.y)+1};
}

function showEvidence(name, html, image=null) {
  el('output-card').hidden = false;
  el('output-title').textContent = name;
  if (evidenceURL) URL.revokeObjectURL(evidenceURL);
  evidenceURL = image ? URL.createObjectURL(image.blob) : null;
  el('output-content').innerHTML = html + (evidenceURL ? `<img class="output-image" alt="${enc(name)}" src="${evidenceURL}">` : '');
}
function clearEvidence() {
  el('output-card').hidden=true;
  if (evidenceURL) URL.revokeObjectURL(evidenceURL);
  evidenceURL = null;
  el('output-content').replaceChildren();
}
function openModal(title, content) {
  el('modal-title').textContent = title;
  el('modal-body').innerHTML = content;
  if (!el('modal').open) el('modal').showModal();
}
function closeModal() { if (el('modal').open) el('modal').close(); }
function textInput(label,id,value='',type='text',accept='') {
  return `<label class="field">${enc(label)}<input id="${id}" type="${type}" value="${enc(value)}" ${accept?`accept="${accept}"`:''}></label>`;
}
function nameDialog(title,defaultValue,action,actionName='Save') {
  openModal(title,textInput('Name','modal-name',defaultValue)+`<div class="flex-buttons">${button(actionName,action,'','primary')}</div>`);
  el('modal-name').focus();
  el('modal-name').select();
}
function importDialog() {
  const canSee = `accept=".png,.tif,.tiff,image/png,image/tiff"`;
  const angles = config.angles.map(a=>`<label class="field">${a}° analyzer<input name="i${a}" type="file" ${canSee}></label>`).join('');
  const patterns = Object.keys(config.mosaic_patterns).map(key=>option(key,key.replaceAll('-', ' · '),'0-45-135-90')).join('');
  openModal('Import capture', `<form id="import-form"><label class="field">Input type<select id="import-type" name="kind"><option value="planes">Four analyzer images</option><option value="mosaic">2 × 2 sensor mosaic</option><option value="package">PolarBench source package</option></select></label><label class="field">Capture name<input name="name" maxlength="100" placeholder="Sample or inspection name"></label><div class="import-fields" id="import-planes">${angles}</div><div class="import-fields" id="import-mosaic" hidden><label class="field">Raw mosaic<input name="mosaic" type="file" ${canSee}></label><label class="field">Tile pattern (upper left, upper right / lower left, lower right)<select name="mosaic_layout">${patterns}</select></label></div><div class="import-fields" id="import-mask"><label class="field">Optional detector inclusion mask<input name="mask" type="file" ${canSee}></label></div><div class="import-fields" id="import-package" hidden><label class="field">Source ZIP<input name="package" type="file" accept=".zip,application/zip"></label></div><div class="flex-buttons"><button class="button primary" type="submit">Import</button></div></form>`);
}
function exportDialog() {
  const items = [['Map PNG','map-png'],['Linear TIFF','map-tiff'],['Evidence ZIP','evidence'],['Inspection worksheet','worksheet'],['Analyzer contact sheet','contact-sheet']];
  openModal('Export current inspection',`<p class="modal-body-copy">The map and worksheet use the current detector window and display settings.</p><div class="flex-buttons">${items.filter(x=>!x[1].includes('tiff') || state.view.layer!=='quality').map(([label,a])=>button(label,a,'','secondary')).join('')}</div>`);
}

const action = {
  'save-default': async()=>{
    project = await forCapture('/default-view','PUT',{...revision(),state});
    toast('Default inspection saved.');
  },
  'crop-view': async()=>{
    if (!activeBox()) throw new Error('Use the Box tool to select a detector area.');
    state.view.crop = copy(activeBox()); target.box=null;
    await updateMap(false); renderPanel();
  },
  'create-crop': async()=>{
    const box = activeBox();
    if (!box) throw new Error('Draw or select a detector box first.');
    const check = await processing('/crop-capture',{box});
    nameDialog('Create cropped capture', `${project.name.slice(0,90)} crop`,'commit-crop','Create');
    const note = document.createElement('p');
    note.className='modal-body-copy';
    note.textContent=`${check.detector_window.width} × ${check.detector_window.height} raw detector pixels · ${check.retained_regions} contained regions`;
    el('modal-body').prepend(note);
  },
  'commit-crop': async()=>{
    const next = await processing('/crop-capture',{box:activeBox(),name:value('modal-name'),mode:'create'});
    closeModal(); await refreshProjects(); await useCapture(next);
    toast('Cropped source capture created.');
  },
  'rename-capture': async()=>nameDialog('Rename capture',project.name,'commit-rename'),
  'commit-rename': async()=>{
    project = await forCapture('','PATCH',{name:value('modal-name'),...revision()});
    closeModal(); renderHeading(); await refreshProjects(); toast('Capture renamed.');
  },
  'delete-capture': async()=>openModal('Delete capture', `<p class="modal-body-copy">Delete ${enc(project.name)} from this workspace, including its regions and saved views?</p><div class="flex-buttons">${button('Delete capture','commit-delete','','danger')}${button('Cancel','close-modal','','secondary')}</div>`),
  'commit-delete': async()=>{
    await forCapture('','DELETE',revision());
    closeModal(); await refreshProjects();
    if (projects.length) await useCapture(null,projects[0].id); else await useCapture(null);
    toast('Capture deleted.');
  },
  'close-modal': async()=>closeModal(),
  'map-png': async()=>downloadBlob(await processing('/download',{output:'png'},true),'polarbench-map.png'),
  'map-tiff': async()=>downloadBlob(await processing('/download',{output:'data'},true),'polarbench-map.tif'),
  'evidence': async()=>downloadBlob(await processing('/download',{output:'evidence'},true),'polarbench-evidence.zip'),
  'contact-sheet': async()=>downloadBlob(await processing('/download',{output:'contact-sheet'},true),'polarbench-contact-sheet.png'),
  'worksheet': async()=>downloadBlob(await processing('/worksheet',{},true),'polarbench-worksheet.png'),
  'apply-register': async()=>{
    const offsets = copy(state.processing.registration);
    for (const a of config.angles.slice(1)) offsets[a]={dx:number(`dx-${a}`),dy:number(`dy-${a}`)};
    const prior = state.processing.registration;
    state.processing.registration = offsets;
    try { await updateMap(true); }
    catch (error) { state.processing.registration = prior; throw error; }
    toast('Offsets applied to this inspection.');
  },
  'suggest-register': async()=>{
    const box = activeBox();
    const radius = box ? Math.max(1,Math.min(16,Math.floor(Math.min(box.width,box.height)/3))) : Math.max(1,Math.min(16,Math.floor(Math.min(project.capture.width,project.capture.height)/3)));
    const outcome = await processing('/registration/suggest',{radius,box});
    const table = [];
    for (const [angle,item] of Object.entries(outcome.channels)) {
      if (item.usable) {
        el(`dx-${angle}`).value = item.offset.dx;
        el(`dy-${angle}`).value = item.offset.dy;
      }
      table.push([`${angle}°`, item.offset ? `${item.offset.dx}, ${item.offset.dy}` : '—',fmt(item.correlation),item.usable?'Proposed':'Review manually']);
    }
    showEvidence('Registration suggestions',detailTable(['Analyzer','Shift','Correlation','Status'],table));
    toast('Review the suggestions, then apply shifts.');
  },
  'review-register': async()=>{
    const options={angle:45,box:activeBox(),mode:'overlay'};
    const [data,image] = await Promise.all([processing('/registration/review',options),processing('/registration/review',{...options,output:'png'},true)]);
    showEvidence('0° and 45° alignment',metrics([['Common pixels',fmt(data.support.common,0)],['Texture correlation',fmt(data.texture_agreement.correlation)],['Shift',`${data.shift.dx}, ${data.shift.dy}`]]),image);
  },
  'apply-calibration': async()=>{
    const settings = copy(state.processing.calibration);
    settings.use_dark=checked('use-dark'); settings.use_flat=checked('use-flat');
    for (const a of config.angles) settings.channels[a]={black:number(`black-${a}`),gain:number(`gain-${a}`)};
    const prior = state.processing.calibration;
    state.processing.calibration=settings;
    try { await updateMap(true); }
    catch (error) { state.processing.calibration=prior; throw error; }
    toast('Intensity correction applied.');
  },
  'sample-reference': async(button)=>{
    project = await forCapture(`/references/${button.dataset.kind}/demo`,'POST',revision());
    renderPanel(); await updateMap(true); toast('Sample reference attached.');
  },
  'upload-reference': async(button)=>{
    const kind=button.dataset.kind;
    const data=uploadFile(`ref-${kind}`);
    project=await forCapture(`/references/${kind}`,'PUT',data);
    renderPanel(); await updateMap(true); toast('Reference uploaded.');
  },
  'clear-reference': async(button)=>{
    const kind=button.dataset.kind;
    project=await forCapture(`/references/${kind}`,'DELETE',revision());
    state.processing.calibration[`use_${kind}`]=false;
    renderPanel(); await updateMap(true); toast('Reference cleared.');
  },
  'set-flat-box': async()=>{
    if (!activeBox()) throw new Error('Draw a normalization box on the detector first.');
    project=await forCapture('/references/flat','PATCH',{box:activeBox(),...revision()});
    renderPanel(); await updateMap(true); toast('Flat normalization box saved.');
  },
  'reference-study': async(button)=>{
    const params={kind:button.dataset.kind,box:activeBox()};
    const [r,img]=await Promise.all([processing('/references/review',params),processing('/references/review',{...params,output:'png'},true)]);
    showEvidence(`${r.kind==='dark'?'Dark':'Flat'} reference`,metrics([['Usable samples',fmt(r.usable,0)],['Median',fmt(r.signal?.median,4)],['Median gain',fmt(r.gain?.median,4)]]),img);
  },
  'neutral-balance': async()=>{
    if (!activeBox()) throw new Error('Draw a detector box on a neutral diffuse target first.');
    const r=await processing('/references/neutral-balance',{geometry:{shape:'box',box:activeBox()}});
    if (!r.available) throw new Error('Estimated gains fall outside the supported range.');
    for (const [a,c] of Object.entries(r.channels)) el(`gain-${a}`).value=c.after;
    showEvidence('Neutral channel balance',metrics([['Common samples',fmt(r.common_samples,0)],['Median degree',fmt(r.predicted_median_degree,4)],['Anchor',fmt(r.target,4)]]));
    toast('Gain estimates filled in. Apply correction to inspect them.');
  },
  'set-scale': async()=>{
    project=await forCapture('/scale','PUT',{...revision(),unit:value('scale-unit'),pitch_x:number('scale-x'),pitch_y:number('scale-y')});
    toast('Detector scale saved.');
  },
  'clear-scale': async()=>{
    project=await forCapture('/scale','DELETE',revision()); renderPanel(); toast('Detector scale cleared.');
  },
  'mosaic-review': async()=>{
    const box=activeBox();
    const [r,img]=await Promise.all([processing('/mosaic-review',{box}),processing('/mosaic-review',{box,output:'png'},true)]);
    const rows=r.lanes.map(c=>[`${c.angle}°`,`${c.position.row}, ${c.position.column}`,fmt(c.native_valid,0),fmt(c.measurement?.median)]);
    showEvidence('Native sensor lanes',detailTable(['Angle','Row, col','Usable','Median code'],rows),img);
  },
  'measure-region': async()=>{
    regionId=value('measure-region')||'';
    const params={...regionSelection(),layer:value('measure-layer')};
    const [r,img]=await Promise.all([processing('/measure',params),processing('/measure',{...params,output:'png'},true)]);
    showEvidence(r.region,metrics([['Usable pixels',fmt(r.layer_usable,0)],['Median',fmt(r.linear?.percentiles.p50)],['Mean axis',r.direction?.mean_axis==null?'—':`${fmt(r.direction.mean_axis,1)}°`],['Coherence',fmt(r.direction?.coherence)]]),img);
    redrawSelection();
  },
  'use-region': async(button)=>{
    regionId=button.dataset.id;
    target.box=null;
    renderPanel(); redrawSelection();
  },
  'save-region': async()=>{
    if (!target.box) throw new Error('Use Box to draw a detector area first.');
    nameDialog('Save detector region',`Region ${project.regions.length+1}`,'commit-region');
  },
  'commit-region': async()=>{
    const result=await forCapture('/regions','POST',{name:value('modal-name'),geometry:{shape:'box',box:target.box},...revision()});
    closeModal(); target.box=null; regionId=result.region.id; await reloadRecord(); toast('Detector region saved.');
  },
  'remove-region': async(button)=>{
    await forCapture(`/regions/${button.dataset.id}`,'DELETE',revision());
    if (regionId===button.dataset.id) regionId='';
    await reloadRecord(); toast('Detector region removed.');
  },
  'region-table': async()=>{
    downloadBlob(await processing('/measure/table',{layer:value('measure-layer'),output:'csv'},true),'polarbench-regions.csv');
  },
  'profile': async()=>{
    const id=value('profile-id');
    if (!target.line && !id) throw new Error('Use Line to draw a scan or select one of the saved lines.');
    const coordinates=id?{profile_id:id}:{geometry:target.line};
    const opts={...coordinates,interpolation:value('profile-interp'),layer:value('measure-layer')};
    const [r,img]=await Promise.all([processing('/scan-line',opts),processing('/scan-line',{...opts,output:'png'},true)]);
    showEvidence(r.name,metrics([['Usable samples',fmt(r.valid_samples,0)],['Gaps',fmt(r.gaps.length,0)],['Distance',`${fmt(r.length,1)} px`]]),img);
  },
  'save-profile': async()=>{
    if (!target.line) throw new Error('Draw a scan line before saving.');
    nameDialog('Save scan line',`Line ${project.profiles.length+1}`,'commit-profile');
  },
  'commit-profile': async()=>{
    await forCapture('/profiles','POST',{name:value('modal-name'),geometry:target.line,...revision()});
    closeModal(); await reloadRecord(); toast('Scan line saved.');
  },
  'orientation-grid': async()=>{
    const opts={box:activeBox(),tile:number('grid-tile')};
    const [r,img]=await Promise.all([processing('/orientation-grid',opts),processing('/orientation-grid',{...opts,output:'png'},true)]);
    showEvidence('Orientation grid',metrics([['Columns',fmt(r.columns,0)],['Rows',fmt(r.rows,0)],['Directional',fmt(r.usable_cells,0)]]),img);
  },
  'bands': async()=>{
    const opts={...regionSelection(),step:number('band-step'),layer:value('measure-layer')==='direction'?'intensity':value('measure-layer')};
    const [r,img]=await Promise.all([processing('/bands',opts),processing('/bands',{...opts,output:'png'},true)]);
    showEvidence('Detector band summary',metrics([['Bands',fmt(r.bands.length,0)],['Usable bands',fmt(r.usable_bands,0)],['Band thickness',`${r.step} px`]]),img);
  },
  'markers': async()=>{
    const opts={...regionSelection(),layer:value('measure-layer')==='direction'?'intensity':value('measure-layer')};
    const [r,img]=await Promise.all([processing('/signal-markers',opts),processing('/signal-markers',{...opts,output:'png'},true)]);
    showEvidence('Signal maxima',detailTable(['Marker','x','y','Value'],r.markers.map(m=>[m.id,m.x,m.y,fmt(m.value,5)])),img);
  },
  'extent': async()=>{
    const r=await processing('/spatial-measurement',target.line?{line:target.line}:regionSelection());
    const items = r.length==null ? [['Width',`${fmt(r.width)} ${r.unit}`],['Height',`${fmt(r.height)} ${r.unit}`],['Area',`${fmt(r.area)} ${r.area_unit}`]] : [['Length',`${fmt(r.length)} ${r.unit}`],['Detector',`${fmt(r.detector_length)} px`]];
    showEvidence('Spatial measurement',metrics(items));
  },
  'sweep': async()=>displaySweep('curve'),
  'sweep-sheet': async()=>displaySweep('png'),
  'apply-quality': async()=>{
    const prior=state.processing.limits;
    state.processing.limits={...prior,min_intensity:number('limit-dark'),max_pair_error:number('limit-closure'),saturation:number('limit-clipped'),min_direction:number('limit-direction'),max_degree:number('limit-physical'),mask_threshold:number('limit-mask')};
    try { await updateMap(true); }
    catch (error) {state.processing.limits=prior; throw error;}
    toast('Quality thresholds updated.');
  },
  'quality-map': async()=>{changeLayer('quality'); await updateMap(true);},
  'quality-focus': async(button)=>{changeLayer('quality'); state.view.quality_focus=button.dataset.kind; await updateMap(false);},
  'quality-patches': async()=>{
    const opts={issue:value('quality-issue'),box:activeBox()};
    const [r,img]=await Promise.all([processing('/quality/patches',opts),processing('/quality/patches',{...opts,output:'png'},true)]);
    const out=detailTable(['Patch','Detector x,y','Pixels'],r.patches.slice(0,12).map((p,i)=>[i+1,`${p.box.x}, ${p.box.y}`,fmt(p.pixels,0)]));
    showEvidence(`${reasonName(r.issue)} patches`,metrics([['All patches',r.all_components],['Qualified',r.qualified_components],['Shown',r.shown]])+out,img);
  },
  'pair-evidence': async()=>{
    const opts={box:activeBox()};
    const [r,img]=await Promise.all([processing('/pair-evidence',opts),processing('/pair-evidence',{...opts,output:'png'},true)]);
    showEvidence('Opposite analyzer pair support',metrics([['Both',fmt(r.coverage.both,0)],['0° + 90° only',fmt(r.coverage.first_only,0)],['45° + 135° only',fmt(r.coverage.second_only,0)]]),img);
  },
  'upload-mask': async()=>{
    project=await forCapture('/mask','PUT',uploadFile('mask-file')); renderPanel(); await updateMap(true); toast('Detector mask uploaded.');
  },
  'clear-mask': async()=>{
    project=await forCapture('/mask','DELETE',revision()); renderPanel(); await updateMap(true); toast('Detector mask cleared.');
  },
  'mask-evidence': async()=>{
    const [r,img]=await Promise.all([processing('/mask/evidence'),processing('/mask/evidence',{output:'png'},true)]);
    showEvidence('Detector mask evidence',metrics([['Excluded',fmt(r.excluded,0)],['Interior holes',fmt(r.interior_holes,0)],['Border components',fmt(r.border_components,0)]]),img);
  },
  'mask-preview': async()=>{
    const opts={...regionSelection('mask-region'),mode:value('mask-mode')};
    const [r,img]=await Promise.all([processing('/mask/edit',opts),processing('/mask/edit',{...opts,action:'preview'},true)]);
    showEvidence('Mask change preview',metrics([['Current',fmt(r.before,0)],['After edit',fmt(r.after,0)],['Removed',fmt(r.removed,0)]])+legends(r.legend),img);
  },
  'mask-apply': async()=>{
    project=await processing('/mask/edit',{...regionSelection('mask-region'),mode:value('mask-mode'),action:'apply',...revision()});
    renderPanel(); await updateMap(true); toast('Mask edit applied.');
  },
  'compare': async()=>{
    const options=compareSettings();
    const [r,img]=await Promise.all([processing('/compare',options),processing('/compare',{...options,output:'png'},true)]);
    const figures=metrics([['Shared pixels',fmt(r.shared_pixels,0)],['Mean abs change',fmt(r.statistics?.absolute_mean,4)],['Threshold crossings',fmt(r.statistics?.above_threshold,0)]]);
    showEvidence(`${r.current.name} − ${r.reference.name}`,figures+legends(r.legend),img);
  },
  'comparison-export': async()=>downloadBlob(await processing('/compare',{...compareSettings(),output:'evidence'},true),'polarbench-comparison.zip'),
  'commit-saved': async()=>{
    await forCapture('/views','POST',{name:value('modal-name'),state,...revision()});
    closeModal(); await reloadRecord(); setTab('views'); toast('Inspection view saved.');
  },
  'show-saved': async(button)=>{
    const saved=project.sessions.find(x=>x.id===button.dataset.id);
    const image=await forCapture(`/views/${saved.id}/preview`,'GET',undefined,true);
    analyzedSequence++;
    mapMode='replay';
    el('view-caption').textContent='SAVED INSPECTION';
    el('frame-label').textContent=saved.name;
    displayedUrl(image.blob); applyViewport(null);
    el('legend-card').innerHTML=`<div class="legend-label"><span>Saved ${enc(saved.saved_at.replace('T',' ').slice(0,16))}</span><span>Use Fit to return to the live detector.</span></div>`;
  },
  'apply-saved': async(button)=>{
    project=await forCapture(`/views/${button.dataset.id}/apply`,'POST',revision());
    state=copy(project.defaults); renderPanel(); await updateMap(true); toast('Saved settings restored.');
  },
  'remove-saved': async(button)=>{
    await forCapture(`/views/${button.dataset.id}`,'DELETE',revision()); await reloadRecord(); toast('Saved view removed.');
  },
};

function uploadFile(id) {
  const file=el(id).files[0];
  if (!file) throw new Error('Select an image file first.');
  const data=new FormData();
  data.append('file',file); data.append('revision',project.revision);
  return data;
}
function reasonName(kind) {return config.quality_legend.find(x=>x.key===kind)?.label || kind;}
function compareSettings() {
  return {reference_id:value('compare-reference'),layer:value('compare-layer'),axis:value('compare-axis'),map_style:value('compare-style'),threshold:number('compare-threshold'),limit:number('compare-range'),box:activeBox()};
}
async function displaySweep(output) {
  const params={...regionSelection(),start:number('sweep-start'),stop:number('sweep-stop'),count:number('sweep-count'),signal_level:number('sweep-level')};
  const [r,img]=await Promise.all([processing('/sweep',params),processing('/sweep',{...params,output},true)]);
  showEvidence(output==='curve'?'Virtual analyzer response':'Virtual analyzer contact sheet',metrics([['Angle samples',r.angles.length],['Usable pixels',fmt(r.usable,0)],['Signal threshold',fmt(r.minimum_signal)]]),img);
}

function prepareInput() {
  document.addEventListener('click', async(event)=>{
    const clicked=event.target.closest('button');
    if (!clicked) return;
    const name=clicked.dataset.action;
    const dataset=clicked.dataset;
    try {
      if (name && action[name]) {
        clicked.disabled=true;
        await action[name](clicked);
      } else if (dataset.tab && project) {
        setTab(dataset.tab); redrawSelection();
      } else if (dataset.tool) {
        setTool(dataset.tool);
      } else if (dataset.project) {
        await useCapture(null,dataset.project);
      } else if (dataset.demo || clicked.id==='empty-demo') {
        clicked.disabled=true;
        const demo=dataset.demo || 'coated-panel';
        const next=await request('/api/projects/demo',{method:'POST',data:{demo}});
        await refreshProjects(); await useCapture(next);
        toast('Synthetic capture opened.');
      } else if (clicked.id==='new-capture' || clicked.id==='empty-import') {
        importDialog();
      } else if (clicked.id==='modal-close') {
        closeModal();
      } else if (clicked.id==='clear-output') {
        clearEvidence();
      } else if (clicked.id==='save-view') {
        nameDialog('Save current view',config.layer_options.find(x=>x.id===state.view.layer).label,'commit-saved');
      } else if (clicked.id==='export-view') {
        exportDialog();
      } else if (clicked.id==='fit-detector') {
        state.view.crop=null;
        target={};
        await updateMap(false); renderPanel();
      }
    } catch(error) {
      toast(error.message,true);
    } finally {
      if (name || dataset.demo || clicked.id==='empty-demo') clicked.disabled=false;
    }
  });
  document.addEventListener('submit',async(event)=>{
    if (event.target.id!=='import-form') return;
    event.preventDefault();
    const submitter=event.submitter;
    if (submitter) submitter.disabled=true;
    try {
      const form=new FormData(event.target);
      const rawKind=form.get('kind');
      const active=rawKind==='planes' ? ['name','kind','i0','i45','i90','i135','mask'] : rawKind==='mosaic'?['name','kind','mosaic','mosaic_layout','mask']:['name','package'];
      for (const name of [...form.keys()]) {
        const current=form.get(name);
        if (!active.includes(name) || (current instanceof File && current.size===0)) form.delete(name);
      }
      const endpoint=rawKind==='package'?'/api/projects/source-package':'/api/projects';
      const next=await request(endpoint,{method:'POST',data:form});
      closeModal(); await refreshProjects(); await useCapture(next); toast('Capture imported.');
    } catch(error) { toast(error.message,true); }
    finally { if (submitter) submitter.disabled=false; }
  });
  document.addEventListener('change',async(event)=>{
    const {id}=event.target;
    if (id==='import-type') {
      for (const kind of ['planes','mosaic','package']) el(`import-${kind}`).hidden=kind!==event.target.value;
      el('import-mask').hidden=event.target.value==='package';
      return;
    }
    if (!state) return;
    if (id==='measure-region') {
      regionId=event.target.value;
      if (regionId) target.box=null;
      redrawSelection(); return;
    }
    if (id==='compare-layer') {
      el('compare-range').value=event.target.value==='direction'?30:.35;
      el('compare-threshold').value=event.target.value==='direction'?10:.10;
      return;
    }
    const before=copy(state);
    try {
      let repaint=true, diagnose=false;
      if (id==='map-layer') { changeLayer(value(id)); renderPanel(); }
      else if (id==='map-palette') state.view.palette=value(id);
      else if (id==='display-low') state.view.display.low=number(id);
      else if (id==='display-high') state.view.display.high=number(id);
      else if (id==='display-gamma') state.view.gamma=number(id);
      else if (id==='show-vectors') state.view.vectors=checked(id);
      else if (id==='show-grid') state.view.grid=checked(id);
      else if (id==='zero-angle') { state.processing.zero_angle=number(id); diagnose=true; }
      else if (id==='map-quality-focus') state.view.quality_focus=value(id)||null;
      else if (id==='invalid-style') state.view.invalid_style=value(id);
      else repaint=false;
      if (repaint) await updateMap(diagnose);
    } catch(error) { state=before; renderPanel(); toast(error.message,true); }
  });
  document.addEventListener('input',(event)=>{
    if (event.target.id!=='analyzer-angle') return;
    state.view.analyzer_angle=number('analyzer-angle');
    el('analyzer-readout').textContent=`${fmt(state.view.analyzer_angle,1)}°`;
    clearTimeout(previewTimer);
    previewTimer=setTimeout(()=>updateMap(false).catch(e=>toast(e.message,true)),125);
  });
  el('modal').addEventListener('click',(event)=>{
    if (event.target===el('modal')) {
      const rect=el('modal').getBoundingClientRect();
      if (event.clientX<rect.left || event.clientX>rect.right || event.clientY<rect.top || event.clientY>rect.bottom) closeModal();
    }
  });
}

function pointerControls() {
  MAP.addEventListener('load',markViewport);
  new ResizeObserver(markViewport).observe(MAP);
  OVERLAY.addEventListener('pointerdown',(event)=>{
    if (event.button !== 0) return;
    const point=fromPointer(event);
    if (!point) return;
    drag={tool,origin:point};
    OVERLAY.setPointerCapture(event.pointerId);
  });
  OVERLAY.addEventListener('pointermove',(event)=>{
    const point=fromPointer(event);
    if (!point) return;
    el('pointer-label').textContent=`x ${point.x} · y ${point.y}`;
    if (!drag) return;
    if (drag.tool==='box') target.box=buildBox(drag.origin,point);
    else if (drag.tool==='line') target.line={start:drag.origin,end:point};
    redrawSelection();
  });
  OVERLAY.addEventListener('pointerup',async(event)=>{
    if (!drag) return;
    const point=fromPointer(event);
    const current=drag;
    drag=null;
    if (!point) return;
    if (current.tool==='box') {
      target.box=buildBox(current.origin,point); target.line=null; regionId='';
      if (tab==='measure') renderPanel();
    } else if (current.tool==='line') {
      if (current.origin.x===point.x && current.origin.y===point.y) return;
      target.line={start:current.origin,end:point};
      if (tab==='measure') renderPanel();
    } else if (current.tool==='probe') {
      target.point=point;
      try {
        const p=await processing('/probe',point);
        const status=p.status.length ? p.status.map(reasonName).join(', ') : 'Usable pixel';
        if (el('probe-result')) el('probe-result').textContent=`(${p.x}, ${p.y}) · ${fmt(p.shown_value,4)} · ${status}`;
        const channels=Object.entries(p.raw_planes).map(([a,x])=>[`${a}°`,x?`${x.sensor_x??x.x}, ${x.sensor_y??x.y}`:'—',x?.code??'—',fmt(p.aligned_planes[a].corrected,4)]);
        const health=metrics([['Intensity',fmt(p.combined.intensity,4)],['Linear degree',fmt(p.combined.degree,4)],['Axis',p.combined.direction==null?'—':`${fmt(p.combined.direction,1)}°`]]);
        showEvidence(`Pixel ${p.x}, ${p.y}`,`<div class="output-info">${enc(status)}</div>${health}${detailTable(['Analyzer','Native x,y','Raw code','Corrected'],channels)}`);
      } catch(error) { toast(error.message,true); }
    }
    redrawSelection();
  });
  OVERLAY.addEventListener('pointercancel',()=>{drag=null; redrawSelection();});
}

async function start() {
  try {
    config=await request('/api/configuration');
    prepareInput(); pointerControls();
    await refreshProjects();
    el('connection-text').textContent='Local workspace';
    const initial = new URLSearchParams(location.hash.slice(1)).get('capture');
    const first=projects.find(p=>p.id===initial) || projects[0];
    if (first) await useCapture(null, first.id);
  } catch(error) {
    el('connection-dot').classList.add('warning');
    el('connection-text').textContent='Workspace offline';
    toast(error.message,true);
  }
}
start();
