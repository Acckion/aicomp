"""GT-free encoder routing for a fixed four-region native pixel budget."""
import torch

@torch.no_grad()
def encoder_candidates(decoder,features):
    memory,shapes=decoder._get_encoder_input(features)
    anchors,valid=decoder._generate_anchors(shapes,device=memory.device)
    projected=decoder.enc_output(memory*valid.to(memory.dtype))
    logits=decoder.enc_score_head(projected)
    scores=logits.amax(-1).masked_fill(~valid.squeeze(-1),-torch.inf)
    indices=scores.topk(decoder.num_queries,dim=1).indices
    selected=projected.gather(1,indices[...,None].expand(-1,-1,projected.shape[-1]))
    selected_anchors=anchors.expand(memory.shape[0],-1,-1).gather(1,indices[...,None].expand(-1,-1,4))
    boxes=(decoder.enc_bbox_head(selected)+selected_anchors).sigmoid()
    return boxes,scores.gather(1,indices).sigmoid()

def grid_windows(fraction=.3):
    assert 0<fraction<=1
    starts=torch.linspace(0,1-fraction,4).tolist()
    return [(x,y,x+fraction,y+fraction) for y in starts for x in starts]

@torch.no_grad()
def select_windows(boxes,scores,fraction=.3,budget=4):
    """Greedy marginal candidate coverage; takes model boxes/scores only."""
    assert boxes.ndim==2 and boxes.shape[1]==4 and scores.shape==(len(boxes),)
    windows=torch.tensor(grid_windows(fraction),device=boxes.device)
    centers=boxes[:,:2]
    inside=((centers[None]>=windows[:,None,:2])&(centers[None]<=windows[:,None,2:])).all(-1)
    eligible=torch.isfinite(boxes).all(-1)&(boxes[:,2:]>0).all(-1)&(boxes[:,2:].amax(-1)<=.15)&(scores>=.05)
    weights=scores.float()*eligible;covered=torch.zeros(len(boxes),device=boxes.device,dtype=torch.bool)
    chosen=[]
    for _ in range(min(budget,len(windows))):
        utility=(inside.float()*(weights*(~covered))[None]).sum(-1)
        if chosen:utility[chosen]=-torch.inf
        if utility.max()>0:
            index=int(utility.argmax())
        elif chosen:
            grid_centers=(windows[:,:2]+windows[:,2:])/2
            distances=(grid_centers[:,None]-grid_centers[chosen][None]).square().sum(-1).amin(-1)
            distances[chosen]=-torch.inf;index=int(distances.argmax())
        else:index=0
        chosen.append(index);covered|=inside[index]
    return [grid_windows(fraction)[i] for i in chosen]
