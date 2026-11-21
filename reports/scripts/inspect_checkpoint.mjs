// Static metadata reader for local torch.save ZIP / pickle protocol 2 artifacts.
import fs from "node:fs/promises";
import { crc32 } from "node:zlib";
import { pathToFileURL } from "node:url";

// Symbolic pickle interpreter. GLOBAL and REDUCE create descriptors and never
// resolve or invoke constructors. Unsupported opcodes fail closed.
function parsePickle(b) {
  const stack = [], memo = new Map(), mark = Symbol();
  let i = 0;
  function slice(n) { if (i + n > b.length) throw Error("Truncated pickle"); const a=b.subarray(i,i+n); i+=n; return a; }
  function uint(n) { return slice(n).readUIntLE(0,n); }
  function line() { const p=b.indexOf(10,i); if(p<0) throw Error("Unterminated pickle line"); const out=slice(p-i).toString(); i++; return out; }
  function marked() { const p=stack.lastIndexOf(mark); if(p<0) throw Error("Missing MARK"); return stack.splice(p).slice(1); }
  function set(dict, items) { for(let n=0;n<items.length;n+=2) dict[items[n]]=items[n+1]; }
  while(i<b.length) {
    const op=b[i++];
    switch(op) {
      case 0x80: if(uint(1)!==2) throw Error("Expected protocol 2"); break;
      case 0x2e: return stack.pop();
      case 0x28: stack.push(mark); break;
      case 0x7d: stack.push(Object.create(null)); break;
      case 0x5d: case 0x29: stack.push([]); break;
      case 0x71: case 0x72: memo.set(uint(op===0x71?1:4),stack.at(-1)); break;
      case 0x68: case 0x6a: stack.push(memo.get(uint(op===0x68?1:4))); break;
      case 0x58: stack.push(slice(uint(4)).toString()); break;
      case 0x4e: stack.push(null); break;
      case 0x88: stack.push(true); break;
      case 0x89: stack.push(false); break;
      case 0x4b: stack.push(uint(1)); break;
      case 0x4d: stack.push(uint(2)); break;
      case 0x4a: stack.push(slice(4).readInt32LE()); break;
      case 0x47: stack.push(slice(8).readDoubleBE()); break;
      case 0x8a: { const bytes=slice(uint(1)); let n=0n; for(let p=bytes.length-1;p>=0;p--) n=(n<<8n)|BigInt(bytes[p]); if(bytes.at(-1)&128)n-=1n<<BigInt(8*bytes.length); stack.push(Number.isSafeInteger(Number(n))?Number(n):String(n)); break; }
      case 0x75: { const items=marked(); set(stack.at(-1),items); break; }
      case 0x73: { const val=stack.pop(), key=stack.pop(); set(stack.at(-1),[key,val]); break; }
      case 0x65: { const items=marked(); stack.at(-1).push(...items); break; }
      case 0x61: { const val=stack.pop(); stack.at(-1).push(val); break; }
      case 0x74: stack.push(marked()); break;
      case 0x85: case 0x86: case 0x87: stack.push(stack.splice(-(op-0x84))); break;
      case 0x63: stack.push({ global: [line(),line()] }); break;
      case 0x52: { const args=stack.pop(), fn=stack.pop(); stack.push({fn,args}); break; }
      case 0x51: stack.push({persistent:stack.pop()}); break;
      case 0x62: { const state=stack.pop(); stack.at(-1).state=state; break; }
      default: throw Error("Unsupported pickle opcode 0x"+op.toString(16)+" at "+(i-1));
    }
  }
  throw Error("Missing STOP");
}

export async function inspect(file, {verifyCRC=false}={}) {
  const f=await fs.open(file);
  try {
    const size=(await f.stat()).size;
    async function read(start,len) { if(start<0||len<0||start+len>size)throw Error("Invalid ZIP range"); const bytes=Buffer.alloc(len); if((await f.read(bytes,0,len,start)).bytesRead!==len)throw Error("Short read"); return bytes; }
    const tail=await read(Math.max(0,size-65557),Math.min(size,65557));
    const e=tail.lastIndexOf(Buffer.from([0x50,0x4b,0x05,0x06]));
    if(e<0||e+22+tail.readUInt16LE(e+20)!==tail.length)throw Error("Invalid ZIP end of central directory");
    const count=tail.readUInt16LE(e+10), clen=tail.readUInt32LE(e+12), coff=tail.readUInt32LE(e+16);
    if(clen>8*1024*1024||coff+clen>size-tail.length+e)throw Error("Unsupported central directory");
    const cd=await read(coff,clen);
    let p=0, storageCount=0, verifiedCRC=0, meta;
    for(let n=0;n<count;n++){
      if(cd.readUInt32LE(p)!==0x02014b50)throw Error("Invalid central entry");
      const namesize=cd.readUInt16LE(p+28), extras=cd.readUInt16LE(p+30), comment=cd.readUInt16LE(p+32);
      const name=cd.subarray(p+46,p+46+namesize).toString();
      const len=cd.readUInt32LE(p+20), off=cd.readUInt32LE(p+42);
      if(off+30+len>coff)throw Error("ZIP record exceeds payload range");
      if(/\/data\/[0-9]+$/.test(name))storageCount++;
      if(verifyCRC){
        if(cd.readUInt16LE(p+10)!==0)throw Error("Unsupported payload compression");
        const head=await read(off,30);
        if(head.readUInt32LE(0)!==0x04034b50)throw Error("Invalid local entry");
        const start=off+30+head.readUInt16LE(26)+head.readUInt16LE(28);
        if(start+len>coff)throw Error("ZIP payload exceeds record range");
        let crc=0;
        for(let offset=0;offset<len;offset+=4*1024*1024)
          crc=crc32(await read(start+offset,Math.min(4*1024*1024,len-offset)),crc);
        if(crc!==cd.readUInt32LE(p+16))throw Error("CRC mismatch for "+name);
        verifiedCRC++;
      }
      if(name.endsWith("/data.pkl")){
        if(cd.readUInt16LE(p+10)!==0||len>8*1024*1024)throw Error("Unsupported metadata compression or size");
        const head=await read(off,30);
        if(head.readUInt32LE(0)!==0x04034b50)throw Error("Invalid local entry");
        meta=parsePickle(await read(off+30+head.readUInt16LE(26)+head.readUInt16LE(28),len));
      }
      p+=46+namesize+extras+comment;
    }
    if(p!==clen||!meta)throw Error("Incomplete archive directory");
    const {schema_version,config,mode,seed,epoch,best_epoch,best_accuracy,history,torch_version,validation_metric,data_manifest,preprocessing,answer_to_idx,scaler_state_dict,scheduler_state_dict}=meta;
    return {file,size,entries:count,verifiedCRC,storageCount,modelKeys:Object.keys(meta.model_state_dict??{}).length,rootKeys:Object.keys(meta),schema_version,config,mode,seed,epoch,best_epoch,best_accuracy,history,torch_version,validation_metric,data_manifest,preprocessing,answer_to_idx,scaler_state_dict,scheduler_state_dict};
  } finally { await f.close(); }
}
if(typeof process!=="undefined"&&process.argv[1]&&pathToFileURL(process.argv[1]).href===import.meta.url){
  const args=process.argv.slice(2);
  for(const file of args.filter(arg=>arg!=="--crc"))
    console.log(JSON.stringify(await inspect(file,{verifyCRC:args.includes("--crc")}),null,2));
}
