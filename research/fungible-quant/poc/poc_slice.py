import json, struct, subprocess, hashlib, time, os
REPO="brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw"; SHARD="model-layer-030.safetensors"
COMMIT="9297b9f1d53af5c67cffa01e30cc071a1ff7144b"
SRC_SHA="a52473451a97970627f752da41dc40fd5bb24c5ff863db9bf185e3591da0a6f4"
URL=f"https://huggingface.co/{REPO}/resolve/{COMMIT}/{SHARD}"
def rget(a,b,out): subprocess.run(["curl","-sL","--max-time","300","-r",f"{a}-{b}",URL,"-o",out],check=True)
rget(0,7,"l8"); hlen=struct.unpack("<Q",open("l8","rb").read())[0]
rget(8,7+hlen,"h.json"); hdr=json.load(open("h.json")); base=8+hlen
names=sorted([k for k in hdr if k.startswith("model.layers.30.mlp.experts.137.")], key=lambda k:hdr[k]["data_offsets"][0])
lo=min(hdr[k]["data_offsets"][0] for k in names); hi=max(hdr[k]["data_offsets"][1] for k in names)
print(f"expert tensors: {len(names)}  span {hi-lo} bytes  contiguous={sum(hdr[k]['data_offsets'][1]-hdr[k]['data_offsets'][0] for k in names)==hi-lo}")
t0=time.time(); rget(base+lo,base+hi-1,"blob"); dt=time.time()-t0
blob=open("blob","rb").read(); print(f"coalesced read: {len(blob)/1e6:.1f} MB in {dt:.1f}s")
# --- write segment file (pure-python safetensors writer) ---
seg_tensors={}; off=0; parts=[]
for k in names:
    a,b=hdr[k]["data_offsets"]; data=blob[a-lo:b-lo]
    seg_tensors[k]={"dtype":hdr[k]["dtype"],"shape":hdr[k]["shape"],"data_offsets":[off,off+len(data)]}
    parts.append(data); off+=len(data)
meta={"__metadata__":{"fq_schema":"fq-segment/1","source_repo":REPO,"source_revision":COMMIT,
 "source_file":SHARD,"source_file_sha256":SRC_SHA,"predicate":"repack-of","k":"3","layer":"30"}}
meta.update(seg_tensors)
hj=json.dumps(meta,separators=(",",":")).encode(); pad=(8-len(hj)%8)%8; hj+=b" "*pad
with open("layer-030.e137.k3.safetensors","wb") as f:
    f.write(struct.pack("<Q",len(hj))); f.write(hj); [f.write(p) for p in parts]
seg_sha=hashlib.sha256(open("layer-030.e137.k3.safetensors","rb").read()).hexdigest()
print("segment written:",os.path.getsize("layer-030.e137.k3.safetensors"),"bytes sha256:",seg_sha[:16])
# --- round-trip verify ---
with open("layer-030.e137.k3.safetensors","rb") as f:
    n=struct.unpack("<Q",f.read(8))[0]; h2=json.loads(f.read(n)); body=f.read()
ok=all(hashlib.sha256(body[h2[k]["data_offsets"][0]:h2[k]["data_offsets"][1]]).hexdigest()==hashlib.sha256(blob[hdr[k]["data_offsets"][0]-lo:hdr[k]["data_offsets"][1]-lo]).hexdigest() for k in names)
print("round-trip byte-identity:",ok)
# --- attestation ---
att={"schema":"fq-attestation/1","predicate":"repack-of","fragment_sha256":seg_sha,
 "materials":{"repo":REPO,"revision":COMMIT,"file":SHARD,"file_sha256":SRC_SHA,
  "tensors":{k:{"offsets":hdr[k]["data_offsets"],"sha256":hashlib.sha256(blob[hdr[k]["data_offsets"][0]-lo:hdr[k]["data_offsets"][1]-lo]).hexdigest()} for k in names[:2]} },
 "note":"per-tensor digests truncated to 2 entries for the committed sample"}
json.dump(att,open("attestation.sample.json","w"),indent=1)
# --- sign (ed25519 via openssl) ---
r=subprocess.run("openssl genpkey -algorithm ed25519 -out poc.key 2>/dev/null && openssl pkeyutl -sign -inkey poc.key -rawin -in attestation.sample.json -out att.sig && openssl pkey -in poc.key -pubout -out poc.pub && openssl pkeyutl -verify -pubin -inkey poc.pub -rawin -in attestation.sample.json -sigfile att.sig",shell=True,capture_output=True,text=True)
print("ed25519 sign+verify:",(r.stdout+r.stderr).strip() or "openssl unavailable")
