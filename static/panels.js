const enc = (value) => String(value ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
export {enc};
export const fmt = (value, decimals = 3) => value == null ? '—' : Number(value).toLocaleString(undefined, {maximumFractionDigits:decimals});
export const option = (value, name, selected) => `<option value="${enc(value)}" ${String(value) === String(selected) ? 'selected' : ''}>${enc(name)}</option>`;
const fields = Object.fromEntries([0,45,90,135].map(a => [a,`i${a}`]));
const select = (id, name, items, selected) => `<label class="field">${enc(name)}<select id="${id}" aria-label="${enc(name)}">${items.map(([v,n]) => option(v,n,selected)).join('')}</select></label>`;
const num = (id, name, value, step='any', min, max) => `<label class="field">${enc(name)}<input id="${id}" type="number" value="${enc(value)}" step="${step}" ${min == null ? '' : `min="${min}"`} ${max == null ? '' : `max="${max}"`}></label>`;
const check = (id, text, value, disabled=false) => `<label class="check-field"><input id="${id}" type="checkbox" ${value?'checked':''} ${disabled?'disabled':''}>${enc(text)}</label>`;
const section = (title, body) => `<section class="panel-section">${title?`<h3>${enc(title)}</h3>`:''}${body}</section>`;
const heading = (name, action) => `<div class="panel-heading"><h2>${enc(name)}</h2>${action ?? ''}</div>`;
const row = (...children) => `<div class="control-row" style="--cols:${Math.min(children.length,4)}">${children.join('')}</div>`;
const actions = (...children) => `<div class="flex-buttons">${children.join('')}</div>`;
export const button = (text, action, extra='', mode='subtle small') => `<button class="button ${mode}" type="button" data-action="${action}" ${extra}>${enc(text)}</button>`;
const regionOptions = (p, text='Entire detector') => [['',text], ...p.regions.map(x => [x.id,x.name])];
const colors = ids => ids.map(x => [x,x.charAt(0).toUpperCase()+x.slice(1)]);
const layerOptions = (config, subset) => config.layer_options.filter(x => !subset || subset.includes(x.id)).map(x => [x.id,x.label]);
const download = (text, url) => `<a class="button subtle small" href="${enc(url)}" download>${enc(text)}</a>`;
const inputFile = (id, label) => `<label class="field">${enc(label)}<input id="${id}" type="file" accept=".png,.tiff,.tif,image/png,image/tiff"></label>`;
const count = (v) => fmt(v,0);

export function metrics(items) {
  return `<div class="metrics">${items.map(([label,value]) => `<div><span>${enc(label)}</span><strong>${enc(value)}</strong></div>`).join('')}</div>`;
}

export function legends(state) {
  if (!state) return '';
  if (state.type === 'category') return `<div class="legend-categories">${state.entries.map(e => `<span title="${enc(e.explanation)}"><i style="background:rgb(${e.color.join(' ')})"></i>${enc(e.label)}</span>`).join('')}</div>`;
  const ticks = state.values;
  const ends = [ticks[0], ticks[Math.floor(ticks.length / 2)], ticks.at(-1)];
  const ribbon = state.colors.map(value => `rgb(${value.join(' ')})`).join(',');
  return `<div class="legend-label"><span>${enc(state.description ?? '')}</span><span>${enc(state.unit)}</span></div><div class="legend-gradient" style="background:linear-gradient(90deg,${ribbon})"></div><div class="legend-ticks">${ends.map(v => `<span>${fmt(v)}</span>`).join('')}</div>`;
}

function mapPanel({config, state, project, projects, report}) {
  const v = state.view;
  const current = config.layer_options.find(x => x.id === v.layer);
  const map = select('map-layer','Map',layerOptions(config),v.layer);
  const zoom = v.crop ? `<p>Detector window ${v.crop.x}, ${v.crop.y} · ${v.crop.width} × ${v.crop.height}</p>` : '';
  const fixed = ['quality','direction'].includes(v.layer);
  const displayed = fixed ? '' : row(num('display-low','Black point',v.display.low,.01),num('display-high','White point',v.display.high,.01))
    + row(select('map-palette','Palette',colors(config.palettes),v.palette), num('display-gamma','Gamma',v.gamma,.05,.35,3));
  const range = v.layer === 'analyzer' ? `<label class="field">Analyzer angle <span id="analyzer-readout">${fmt(v.analyzer_angle,1)}°</span><input id="analyzer-angle" type="range" min="0" max="179.5" value="${enc(v.analyzer_angle)}" step="0.5"></label>` : '';
  const focus = v.layer === 'quality' ? select('map-quality-focus','Highlight', [['','All reasons'],...config.quality_legend.slice(1).map(x => [x.key,x.label])],v.quality_focus) : '';
  const controls = map + range + focus + displayed + row(select('invalid-style','Missing samples', [['transparent','Transparent'],['checker','Checker'],['black','Black']],v.invalid_style),num('zero-angle','Reference angle',state.processing.zero_angle,1,0,180));
  const overlays = row(check('show-vectors','Axis marks',v.vectors),check('show-grid','Detector grid',v.grid));
  const q = report?.quality;
  const health = q ? metrics([['Sampled',count(q.usable)],['Axis usable',count(q.orientation_usable)],['Detector px',count(q.pixels)]]) : '<p>Reading detector coverage…</p>';
  return heading('Map controls') + section('',controls + overlays + actions(button('Save default','save-default')))
    + section('Coverage', health + zoom)
    + recoveryPanel({state, project, projects, report})
    + section('Pixel probe', `<p>Click on the image to inspect calibrated intensity, orientation and the four native detector samples.</p><div id="probe-result" class="help-line"></div>`)
    + section('Output framing',`<p>Draw a box to crop the map or create a separate captured image set.</p>${actions(button('Crop view','crop-view'),button('New capture from box','create-crop'))}`);
}

function recoveryPanel({state, project, projects, report}) {
  const r = state.processing.recovery;
  const sameSize = (projects || []).filter(x => x.id !== project.id
    && x.width === project.capture.width && x.height === project.capture.height);
  const options = [['','Select a same-size capture'], ...sameSize.map(x => [x.id, `${x.name} (${x.width} × ${x.height})`])];
  const picker = sameSize.length
    ? select('recovery-short','Short exposure capture', options, r.short_capture_id || '')
    : '<p>Import the short exposure set as a separate capture with matching detector dimensions to enable fill-in.</p>';
  const fields = row(num('recovery-ratio','Normal / short ×',r.exposure_ratio,.1,1.0001,100000),
    num('recovery-min-short','Min short signal',r.min_short,.001,.001,.5))
    + num('recovery-agreement','Agreement tolerance',r.agreement,.01,.001,1);
  const rec = report?.recovery;
  const summary = rec ? metrics([
      ['Recovered',count(rec.recovered_pixels)],
      ['Short fills',count(rec.short_filled_pixels)],
      ['Conflicts',count(rec.agreement_conflict_pixels)],
    ]) + `<p class="help-line">Source ${enc(rec.short_capture_name)} · ×${fmt(rec.exposure_ratio,2)} exposure · main-only pixels are never replaced.</p>` : '';
  return section('Short-exposure fill',
    `<p>Recover clipped angles from an in-place short exposure set. Pixels need four trusted angles that agree after scaling; mask exclusions stay blank.</p>`
    + check('recovery-enabled','Enable short-exposure fill',r.enabled)
    + picker + fields
    + actions(button('Apply fill','apply-recovery','','primary'),button('Clear fill','clear-recovery'))
    + summary);
}

function sourcePanel({project:p, state, config}) {
  const cap = p.capture;
  const id = p.id;
  const sourceLinks = Object.entries(cap.sources).map(([field,x]) => {
    const result = field === 'mosaic' ? 'Packed detector' : `${field.slice(1)}° analyzer`;
    return `<div class="item"><div><strong>${enc(result)}</strong><small>${enc(x.name)} · ${x.bits}-bit</small></div>${download('Raw',`/api/projects/${id}/sources/${field}`)}</div>`;
  }).join('');
  const offsets = config.angles.slice(1).map(a => row(num(`dx-${a}`,`${a}° Δx`,state.processing.registration[a].dx,1,-160,160),num(`dy-${a}`,`${a}° Δy`,state.processing.registration[a].dy,1,-160,160))).join('');
  const cal = state.processing.calibration;
  const tunings = config.angles.map(a => row(num(`black-${a}`,`${a}° black`,cal.channels[a].black,.005,0,.25),num(`gain-${a}`,`${a}° gain`,cal.channels[a].gain,.02,.25,4))).join('');
  const refs = ['dark','flat'].map(kind => {
    const ref = p.references[kind];
    const example = p.capture.demo ? button('Use sample','sample-reference',`data-kind="${kind}"`) : '';
    const available = ref ? `<p>${enc(ref.file.name)}</p>${actions(download('Raw',`/api/projects/${id}/sources/${kind}`), button('Review','reference-study',`data-kind="${kind}"`),button('Clear','clear-reference',`data-kind="${kind}"`))}` : '';
    return `<details><summary>${kind === 'dark'?'Dark signal':'Flat illumination'}</summary>${available}${inputFile(`ref-${kind}`,'Reference PNG or TIFF')}${actions(button('Upload','upload-reference',`data-kind="${kind}"`),example,kind==='flat'?button('Use box for flat','set-flat-box'):'')}</details>`;
  }).join('');
  const pitch = p.scale ?? {};
  const scale = select('scale-unit','Length unit',[['mm','mm'],['cm','cm'],['um','µm'],['in','in']],pitch.unit || 'mm')
    + row(num('scale-x','Horizontal pitch',pitch.pitch_x ?? 0.1,.0001,0.00000001,1000),num('scale-y','Vertical pitch',pitch.pitch_y ?? 0.1,.0001,0.00000001,1000))
    + actions(button('Set scale','set-scale'),button('Clear','clear-scale'));
  return heading('Source preparation') + section('Original images',`<div class="item-list">${sourceLinks}</div>${actions(button('Contact sheet','contact-sheet'),cap.type==='mosaic'?button('Sensor tiles','mosaic-review'):'',download('Source package',`/api/projects/${id}/source-package`))}`)
    + section('Register analyzers', `<p>0° defines detector coordinates. Inspect the overlay before retaining estimated shifts.</p>${offsets}${actions(button('Apply shifts','apply-register'),button('Suggest shifts','suggest-register'),button('Registration overlay','review-register'))}`)
    + section('Intensity correction',`<p>Dark and flat maps are sampled in native detector coordinates before alignment.</p>${refs}${row(check('use-dark','Apply dark',cal.use_dark,!p.references.dark),check('use-flat','Apply flat',cal.use_flat,!p.references.flat))}<details><summary>Per-angle tuning</summary>${tunings}</details>${actions(button('Apply correction','apply-calibration'),button('Neutral balance','neutral-balance'))}`)
    + section('Detector scale',scale)
    + section('Capture details',actions(button('Rename','rename-capture'),button('Delete capture','delete-capture','','danger small')));
}

function measurePanel({project:p, config, state, selection, regionId}) {
  const selected = selection?.box;
  const selectedLine = selection?.line;
  const label = selected ? `${selected.x}, ${selected.y} · ${selected.width} × ${selected.height}` : 'No box drawn';
  const scan = selectedLine ? `${selectedLine.start.x}, ${selectedLine.start.y} to ${selectedLine.end.x}, ${selectedLine.end.y}` : 'No line drawn';
  const regionItems = p.regions.length ? `<div class="item-list">${p.regions.map(x => `<div class="item"><i style="background:${enc(x.color)}"></i><div><strong>${enc(x.name)}</strong><small>${enc(x.geometry.shape)} · ${x.geometry.box.width} × ${x.geometry.box.height}</small></div>${button('Use','use-region',`data-id="${x.id}"`)}<button class="quiet" data-action="remove-region" data-id="${x.id}" aria-label="Delete ${enc(x.name)}">×</button></div>`).join('')}</div>` : '';
  return heading('Detector measurements')
    + section('Region',`<div class="help-line"><span>Current box</span><code>${enc(label)}</code></div>${select('measure-region','Scope',regionOptions(p,'Live box or entire detector'),regionId)}${select('measure-layer','Signal',layerOptions(config,config.scalar_layers),state.view.layer==='quality'?'intensity':state.view.layer)}${actions(button('Measure region','measure-region'),button('Save region','save-region'),button('Table CSV','region-table'))}${regionItems}`)
    + section('Scan line',`<div class="help-line"><span>Current line</span><code>${enc(scan)}</code></div>${select('profile-id','Scan line',[['','Live line'],...p.profiles.map(x=>[x.id,x.name])],'')}${select('profile-interp','Sample method',[['bilinear','Bilinear'],['nearest','Nearest']], 'bilinear')}${actions(button('Plot scan line','profile'),button('Save line','save-profile'))}`)
    + section('Spatial summaries', `${row(num('grid-tile','Cell size',40,1,4,600),num('band-step','Band thickness',8,1,1,160))}${actions(button('Orientation grid','orientation-grid'),button('Detector bands','bands'),button('Signal markers','markers'),button('Measure extent','extent'))}`)
    + section('Analyzer sweep',`${row(num('sweep-start','Start °',0,1,0,180),num('sweep-stop','Stop °',165,1,0,180))}${row(num('sweep-count','Angles',12,1,2,36),num('sweep-level','Signal threshold',.2,.01,0,8))}${actions(button('Response curve','sweep'),button('Angle contact sheet','sweep-sheet'))}`);
}

function qualityPanel({config,state,report,project}) {
  const l = state.processing.limits;
  const current = report?.quality;
  const reasonOptions = config.quality_legend.slice(1).map(x => [x.key,x.label]);
  const reasonRows = current ? `<div class="item-list">${config.quality_legend.slice(1).map(e => `<div class="item"><i style="background:rgb(${e.color.join(' ')})"></i><div><strong>${enc(e.label)}</strong><small>${count(current.issues[e.key])} pixels</small></div>${button('Show','quality-focus',`data-kind="${e.key}"`)}</div>`).join('')}</div>` : '';
  const activeMask = project.mask ? `<p>Using ${enc(project.mask.name)}.</p>` : '<p>All detector positions are included before acquisition checks.</p>';
  return heading('Quality and exclusions')
    + section('Thresholds',`${row(num('limit-dark','Min intensity',l.min_intensity,.005,.0001,4),num('limit-closure','Max pair error',l.max_pair_error,.01,.001,1))}${row(num('limit-clipped','Raw saturation',l.saturation,.005,.4,1),num('limit-direction','Min axis degree',l.min_direction,.005,.001,1))}${row(num('limit-physical','Max degree',l.max_degree,.01,1,1.3),num('limit-mask','Mask cutoff',l.mask_threshold,1,1,255))}${actions(button('Apply thresholds','apply-quality'),button('Show quality map','quality-map'))}`)
    + section('Connected issue patches',select('quality-issue','Reason',reasonOptions,'alpha') + actions(button('Locate patches','quality-patches'),button('Analyzer pairs','pair-evidence')) + reasonRows)
    + section('Detector mask',activeMask + inputFile('mask-file','Mask PNG or TIFF') + actions(button('Upload mask','upload-mask'),button('Mask evidence','mask-evidence'),button('Clear mask','clear-mask')) + select('mask-mode','Edit from region',[['exclude','Exclude selected pixels'],['keep','Keep selected only'],['restore','Restore selected pixels'],['intersect','Intersect with selection']],'exclude') + select('mask-region','Selection',regionOptions(project,'Current box'),'') + actions(button('Preview edit','mask-preview'),button('Apply edit','mask-apply')));
}

function comparePanel({project:p, projects, state}) {
  const compatible = projects.filter(x => x.id !== p.id && x.width === p.capture.width && x.height === p.capture.height);
  if (!compatible.length) return heading('Capture comparison') + section('Reference capture','<p>Import or create a second capture with matching detector dimensions to map signed changes and overlapping coverage.</p>');
  return heading('Capture comparison') + section('Reference capture', select('compare-reference','Compare against',compatible.map(x=>[x.id,x.name]),compatible[0].id)
    + select('compare-layer','Difference', [['degree','Linear degree'],['direction','Orientation'],['intensity','Total intensity'],['polarized','Polarized intensity'],['pair-error','Pair closure ratio']],state.view.layer)
    + select('compare-axis','Axis',[['detector','Shared detector directions'],['selected-reference','Use each selected reference axis']],'detector')
    + select('compare-style','Show',[['difference','Signed difference'],['support','Shared coverage'],['exceeds-threshold','Threshold crossings']],'difference')
    + row(num('compare-threshold','Change threshold',.10,.01,0,90),num('compare-range','Display range',.35,.01,.0001,90))
    + actions(button('Compare captures','compare'),button('Export comparison','comparison-export')));
}

function viewsPanel({project}) {
  const repo = `/api/projects/${project.id}/views`;
  return heading('Saved inspections') + section('View history', !project.sessions.length ? '<p>Save the current map, measurement settings and detector references. Reopen the original state even after replacing a reference.</p>' : `<div class="item-list">${project.sessions.map(v => `<div class="item"><div><strong>${enc(v.name)}</strong><small>${enc(v.saved_at.replace('T',' ').slice(0,16))}${v.current_sources_match?'':' · archived sources'}</small></div>${button('Open','show-saved',`data-id="${v.id}"`)}<button class="quiet" data-action="remove-saved" data-id="${v.id}" aria-label="Delete saved ${enc(v.name)}">×</button></div>${actions(v.current_sources_match?button('Use settings','apply-saved',`data-id="${v.id}"`):'',download('Evidence ZIP',`${repo}/${v.id}/evidence`))}`).join('')}</div>`);
}

export const PANEL = {map:mapPanel,sources:sourcePanel,measure:measurePanel,quality:qualityPanel,compare:comparePanel,views:viewsPanel};

export function detailTable(headers, rows) {
  return `<table class="detail-table"><thead><tr>${headers.map(x=>`<th>${enc(x)}</th>`).join('')}</tr></thead><tbody>${rows.map(row=>`<tr>${row.map(x=>`<td>${enc(x)}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
}
