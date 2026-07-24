import os,sys,time,torch
sys.path.insert(0,"/work/gdn_kernel")
import chunked_gdn_nki as nki
import transformers.models.qwen3_5.modeling_qwen3_5 as M
_o=M.torch_chunk_gated_delta_rule
def _p(query,key,value,g,beta,chunk_size=64,initial_state=None,output_final_state=False,use_qk_l2norm_in_kernel=False,**kw):
    try:
        out=nki.gdn_chunked_nki(query,key,value,g,beta,chunk_size=chunk_size,use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel)
        return (out,None)
    except Exception as e:
        print("fallback",repr(e),flush=True)
        return _o(query,key,value,g,beta,chunk_size=64,initial_state=initial_state,output_final_state=output_final_state,use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,**kw)
M.torch_chunk_gated_delta_rule=_p
rank=int(os.environ.get("RANK","0")); W=int(os.environ.get("WORLD_SIZE","1"))
import torch_neuronx, torch_neuronx.distributed
import torch.distributed as dist
dist.init_process_group(backend="neuron")
dev=torch.device("neuron")
from transformers import AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
m=AutoModelForCausalLM.from_pretrained("/work/Qwen3.5-4B",trust_remote_code=True,dtype=torch.bfloat16,attn_implementation="eager")
m.config.use_cache=False
lc=LoraConfig(r=16,lora_alpha=32,lora_dropout=0.0,bias="none",target_modules=["q_proj","k_proj","v_proj","o_proj"],task_type="CAUSAL_LM")
m=get_peft_model(m,lc); m=m.to(dev).to(torch.bfloat16)
m=FSDP(m,sharding_strategy=ShardingStrategy.FULL_SHARD,use_orig_params=True,device_id=dev); m.train()
cfg=AutoConfig.from_pretrained("/work/Qwen3.5-4B",trust_remote_code=True); V=getattr(cfg,"vocab_size",None) or cfg.text_config.vocab_size
opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4)
torch.manual_seed(rank); ids=torch.randint(0,V,(1,512),device=dev); lab=ids.clone(); ts=[]
for s in range(6):
    t=time.time(); opt.zero_grad(); out=m(input_ids=ids,labels=lab); loss=out.loss; loss.backward(); opt.step(); dt=time.time()-t
    if rank==0: l=float(loss.detach().float().cpu()); ts.append((s,dt,l)); print(f"[step {s}] {dt:.2f}s loss={l:.4f} fin={l==l}",flush=True)
if rank==0:
    warm=[t for s,t,_ in ts if s>0]; avg=sum(warm)/len(warm); tok=512*W
    print(f"=== INTEGRATED RESULT === {avg:.3f}s/step agg_tok/s={tok/avg:.1f} world={W} ALLFIN={all(l==l for _,_,l in ts)}",flush=True)
print("INTEGRATED_DONE",flush=True)
