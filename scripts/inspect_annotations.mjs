// Independent annotation inspection fallback. Does not run Python tests or decode images.
import * as fs from 'node:fs/promises';
import * as crypto from 'node:crypto';
import path from 'node:path';
const root = process.cwd();
const source = process.argv[2];
if (!source) throw Error('Usage: node scripts/inspect_annotations.mjs <source.csv>');
const inventory = JSON.parse(await fs.readFile(path.join(root,'outputs/results/archive_inventory.json'),'utf8'));
const revision = inventory.revision;
const entries = inventory.entries;
const parseCSV = (text)=>{var rows=[],row=[],field='',quoted=false;for(var i=0;i<text.length;i++){var c=text[i];if(c==='"'){if(quoted&&text[i+1]==='"'){field+='"';i++;}else quoted=!quoted;}else if(c===','&&!quoted){row.push(field);field='';}else if((c==='\n'||c==='\r')&&!quoted){if(c==='\r'&&text[i+1]==='\n')i++;row.push(field);if(row.some(x=>x!==''))rows.push(row);row=[];field='';}else field+=c;}if(field||row.length){row.push(field);rows.push(row);}return rows;};
const quantile = (a,p)=>{var n=(a.length-1)*p,lo=Math.floor(n),hi=Math.ceil(n);return a[lo]+(a[hi]-a[lo])*(n-lo);};
const stats = (rows)=>{var lengths=rows.map(r=>r.answer.trim().split(/\s+/).length).sort((a,b)=>a-b);var ids=new Set(rows.map(r=>r.image_id));var splits={};for(var r of rows){splits[r.split]??={qa:0,ids:new Set()};splits[r.split].qa++;splits[r.split].ids.add(r.image_id);}return {qa_pairs:rows.length,images:ids.size,unique_answers:new Set(rows.map(r=>r.answer)).size,answer_length:{median:quantile(lengths,.5),p95:quantile(lengths,.95),max:lengths.at(-1)},splits:Object.fromEntries(Object.entries(splits).map(([s,v])=>[s,{qa_pairs:v.qa,images:v.ids.size}]))};};
const data = await fs.readFile(source);
const table = parseCSV(data.toString('utf8'));
const columns = table.shift();
const records = table.map(r=>Object.fromEntries(columns.map((c,i)=>[c,r[i]])));
const factoidTypes = ["Detailed Verification","General Health Assessment","Plant Species Identification","Visual Attribute Grounding"];
const filtered = records.filter(r=>factoidTypes.includes(r.question_type));
const trainIds = new Set(records.filter(r=>r.split==='train').map(r=>r.image_id));
const testIds = new Set(records.filter(r=>r.split==='test').map(r=>r.image_id));
const archiveImages = entries.filter(e=>/\.(jpg|jpeg|png)$/i.test(e.name));
const imageNames = new Set(archiveImages.map(e=>e.name.split('/').at(-1).replace(/\.(jpg|jpeg|png)$/i,'')));
const report = {revision, execution_engine:'Node.js independent inspection; Python unavailable', columns,
  raw:stats(records), factoid:stats(filtered),
  question_types:Object.fromEntries([...new Set(records.map(r=>r.question_type))].sort().map(t=>[t,stats(records.filter(r=>r.question_type===t))])),
  train_test_overlap:[...trainIds].filter(id=>testIds.has(id)).length,
  archive_image_files:archiveImages.length,
  annotation_images_missing_from_archive:[...new Set(records.map(r=>r.image_id))].filter(id=>!imageNames.has(id.replace(/\.(jpg|jpeg|png)$/i,''))).length,
  images_without_qa:archiveImages.length-stats(records).images,
  source_sha256:crypto.createHash('sha256').update(data).digest('hex')};
await fs.writeFile(path.join(root,'outputs/results/observed_numbers.json'),JSON.stringify(report,null,2));
console.log(JSON.stringify(report,null,2));
