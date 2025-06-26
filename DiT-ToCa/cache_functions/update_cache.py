import torch
import torch.nn.functional as F
def update_cache(fresh_indices, fresh_tokens, cache_dic, current, fresh_attn_map=None):
    '''
    Update the cache with the fresh tokens.
    '''
    step = current['step']
    layer = current['layer']
    module = current['module']
    
    # Update the cached tokens at the positions
    if module == 'attn': 
        # this branch is not used in the final version, but if you explore the partial fresh strategy of attention, it works.
        indices = fresh_indices.sort(dim=1, descending=False)[0]
        
        cache_dic['attn_map'][-1][layer].scatter_(dim=1, index=indices.unsqueeze(-1).expand(-1, -1, fresh_attn_map.shape[-1]), src=fresh_attn_map)
    elif module == 'mlp':
        indices = fresh_indices
    
    if cache_dic['use_ResCa'] == True and fresh_tokens.shape[1] > 0:
        # import ipdb
        # ipdb.set_trace()  # Debugging point, remove in production

        cache = cache_dic['cache'][-1][layer][module]  # 原地更新
        batch_size, seq_len, dim = cache.shape
        positions = torch.arange(seq_len, device=cache.device).unsqueeze(0)  # [1, T]

        # 1. 取出 fresh 部分，算残差
        idx_fresh_exp = indices.unsqueeze(-1).expand(-1, -1, dim)  # [B, F, D]
        cached_fresh  = cache.gather(dim=1, index=idx_fresh_exp)    # [B, F, D]
        residuals     = fresh_tokens - cached_fresh                 # [B, F, D]

        # 2. 用 boolean mask 快速拿到 un-fresh 下标
        mask_fresh = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=cache.device)
        mask_fresh.scatter_(1, indices, True)  # fresh 标 True
        # idx_unfresh: [B, U]
        idx_unfresh = positions.expand(batch_size, -1)[~mask_fresh].view(batch_size, -1)
        idx_unfresh_exp = idx_unfresh.unsqueeze(-1).expand(-1, -1, dim)  # [B, U, D]

        # 3. 计算 un-fresh tokens、相似度+argmax
        un_fresh = cache.gather(dim=1, index=idx_unfresh_exp)           # [B, U, D]
        # 如果硬件支持，可先转半精度：
        un_fresh = un_fresh.half(); fresh_hp = fresh_tokens.half()
        sim_matrix = torch.bmm(un_fresh, fresh_hp.transpose(1, 2))
        # sim_matrix = torch.bmm(un_fresh, fresh_tokens.transpose(1, 2))   # [B, U, F]
        best_idx = sim_matrix.argmax(dim=2, keepdim=True)               # [B, U, 1]
        best_idx_exp = best_idx.expand(-1, -1, dim)                     # [B, U, D]
        residuals_un = residuals.gather(dim=1, index=best_idx_exp)      # [B, U, D]

        # 4. 原地加残差， fresh 和 un-fresh 一次搞定
        cache.scatter_add_(dim=1, index=idx_fresh_exp,   src=residuals)
        cache.scatter_add_(dim=1, index=idx_unfresh_exp, src=residuals_un)

        # 这样 cache 就被更新成 new_cache 了
        cache_dic['cache'][-1][layer][module] = cache

       
    else:
        cache_dic['cache'][-1][layer][module].scatter_(dim=1, index=indices.unsqueeze(-1).expand(-1, -1, fresh_tokens.shape[-1]), src=fresh_tokens)
    

    

    
#! ResCa 正确代码存档
# # 1. 取出原始缓存张量
# cache = cache_dic['cache'][-1][layer][module]  # [B, T, D]
# batch_size, seq_len, dim = cache.shape

# # 2. 计算 fresh tokens 的残差
# idx_fresh = indices.unsqueeze(-1).expand(-1, -1, dim)  # [B, F, D]
# cached_fresh = cache.gather(dim=1, index=idx_fresh)    # [B, F, D]
# residuals    = fresh_tokens - cached_fresh             # [B, F, D]

# # 3. 计算所有 un-fresh token 在序列维度上的位置 idx_unfresh
# mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=cache.device)
# mask.scatter_(dim=1, index=indices, value=False)       # fresh 位置设为 False
# positions   = torch.arange(seq_len, device=cache.device) \
#                 .unsqueeze(0).expand(batch_size, -1)  # [B, T]
# idx_unfresh = positions[mask].view(batch_size, -1)     # [B, U]

# # 4. 直接使用 idx_unfresh 提取 un-fresh tokens
# idx_unfresh_exp = idx_unfresh.unsqueeze(-1).expand(-1, -1, dim)  # [B, U, D]
# un_fresh        = cache.gather(dim=1, index=idx_unfresh_exp)    # [B, U, D]

# # 5. 找到每个 un-fresh token 最相似的 fresh token，并把残差加上
# sim_matrix  = torch.bmm(un_fresh, fresh_tokens.transpose(1, 2))  # [B, U, F]
# best_idx    = sim_matrix.argmax(dim=2)                      # [B, U]
# best_idx_exp = best_idx.unsqueeze(-1).expand(-1, -1, dim)   # [B, U, D]
# res_for_un   = residuals.gather(dim=1, index=best_idx_exp)  # [B, U, D]
# un_fresh_upd = un_fresh + res_for_un                        # [B, U, D]

# # 6. 用 scatter_ 一次性把 fresh 和 un-fresh 都写回缓存
# new_cache = cache.clone()
# new_cache.scatter_(dim=1, index=idx_fresh,      src=fresh_tokens)   # 更新 fresh
# new_cache.scatter_(dim=1, index=idx_unfresh_exp, src=un_fresh_upd)  # 更新 un-fresh

# # 7. 保存回原始数据结构
# cache_dic['cache'][-1][layer][module] = new_cache