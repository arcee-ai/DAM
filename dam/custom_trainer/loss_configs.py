default = {
        "kl":True, # default is True
        "mse":False, # default is False - proposed alternative to kl
        "entropy":False, # default is False - proposed alternative to kl
        "similarity": True, # default is True
        "overlap": False,   # default is False - proposed alternative to similarity
        "l1_l2_reg": False # default is False
    }

loss_configs = {}
loss_configs["default"] = default
loss_configs["kl_sim_reg"] = default.copy().update({"l1_l2_reg": True}) # add regularization
loss_configs["kl"] = default.copy().update({"similarity": False}) # Ablation: remove similarity
loss_configs["mse_sim"] = default.copy().update({"kl": False, 
                                                 "mse": True}) # Alternative to KL
loss_configs["entropy_sim"] = default.copy().update({"kl": False, 
                                                     "entropy": True})  
loss_configs["kl_overlap"] = default.copy().update({"similarity": False,
                                                "overlap": True}) # Alternative to similarity