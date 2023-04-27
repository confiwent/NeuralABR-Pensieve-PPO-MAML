import os
import torch
import numpy as np
import pdb

M_IN_K = 1000.0
BUFFER_NORM_FACTOR = 10.0
EPSILON = 14

def get_opt_space(ob, state, last_bitrate, index, br_versions, rebuf_p, \
        smooth_p, log, video_chunk_sizes, past_errors):
    br_dim = len(br_versions)
    output = np.zeros(br_dim)
    output[0] = 1.0

    ## bandwidth prediction 
    past_bandwidths = ob[0,-5:]
    while past_bandwidths[0] == 0.0:
        past_bandwidths = past_bandwidths[1:]

    bandwidth_sum = 0
    for past_val in past_bandwidths:
        bandwidth_sum += (1/float(past_val))
    harmonic_bandwidth = 1.0/(bandwidth_sum/len(past_bandwidths))
    # future_bandwidth = harmonic_bandwidth

    # future bandwidth prediction
    # divide by 1 + max of last 5 (or up to 5) errors
    max_error = 0
    error_pos = -5
    if ( len(past_errors) < 5 ):
        error_pos = -len(past_errors)
    max_error = float(max(past_errors[error_pos:]))
    future_bandwidth = harmonic_bandwidth/(1+max_error)  # robustMPC here
    # future_bandwidth = harmonic_bandwidth

    ## last quality
    last_q = last_bitrate
    
    ## simulator infos
    # _, _, _, _,  = env.get_env_info()

    start_buffer = state[1,-1] * BUFFER_NORM_FACTOR

    ## Search feasible space 
    for q in range(br_dim):

        output[q] = 1.0

        q_h = q + 1
        if q_h == br_dim:
            break

        ## quality values
        chunk_q_l = np.log(br_versions[q] / \
                        float(br_versions[0])) if log else \
                            br_versions[q] / M_IN_K
        chunk_q_h = np.log(br_versions[q_h] / \
                        float(br_versions[0])) if log else \
                            br_versions[q_h] / M_IN_K
        chunk_q_init = np.log(br_versions[last_q] / \
                        float(br_versions[0])) if log else \
                            br_versions[last_q] / M_IN_K
        
        ## rebuffer time penalities
        download_time_l = (video_chunk_sizes[q][index])/\
            1000000./future_bandwidth # this is MB/MB/s --> seconds
        download_time_h = (video_chunk_sizes[q_h][index])/\
            1000000./future_bandwidth # this is MB/MB/s --> seconds

        rebuffer_time_l = max(download_time_l - start_buffer, 0)
        rebuffer_time_h = max(download_time_h - start_buffer, 0)

        Rebuf_p_l = rebuf_p * rebuffer_time_l
        Rebuf_p_h = rebuf_p * rebuffer_time_h

        # pdb.set_trace()

        ## Criterion for reducing the action space
        if last_q <= q:
            if chunk_q_l - chunk_q_h - \
                rebuf_p * (Rebuf_p_l - Rebuf_p_h) > EPSILON:
                break
        else:
            if (1 + 2 * smooth_p) * (chunk_q_l - chunk_q_h) - \
                rebuf_p * (Rebuf_p_l - Rebuf_p_h) > EPSILON:
                break 


    return output, future_bandwidth

def get_opt_space_vmaf(ob, state, last_quality, index, br_versions, quality_p, rebuf_p, \
        smooth_p, smooth_n, video_chunk_sizes, video_chunk_psnrs, past_errors):
    br_dim = len(br_versions)
    output = np.zeros(br_dim)
    output[0] = 1.0

    ## bandwidth prediction 
    past_bandwidths = ob[0,-5:]
    while past_bandwidths[0] == 0.0:
        past_bandwidths = past_bandwidths[1:]

    bandwidth_sum = 0
    for past_val in past_bandwidths:
        bandwidth_sum += (1/float(past_val))
    harmonic_bandwidth = 1.0/(bandwidth_sum/len(past_bandwidths))
    # future_bandwidth = harmonic_bandwidth

    # future bandwidth prediction
    # divide by 1 + max of last 5 (or up to 5) errors
    max_error = 0
    error_pos = -5
    if ( len(past_errors) < 5 ):
        error_pos = -len(past_errors)
    max_error = float(max(past_errors[error_pos:]))
    future_bandwidth = harmonic_bandwidth/(1+max_error)  # robustMPC here
    # future_bandwidth = harmonic_bandwidth

    ## last quality
    last_q = last_quality
    
    ## simulator infos
    # _, _, _, _,  = env.get_env_info()

    start_buffer = state[1,-1] * BUFFER_NORM_FACTOR

    ## Search feasible space 
    for q in range(br_dim):

        output[q] = 1.0

        q_h = q + 1
        if q_h == br_dim:
            break

        ## quality values
        chunk_q_l = video_chunk_psnrs[q][index]
        chunk_q_h = video_chunk_psnrs[q_h][index]
        
        ## rebuffer time penalities
        download_time_l = (video_chunk_sizes[q][index])/\
            1000000./future_bandwidth # this is MB/MB/s --> seconds
        download_time_h = (video_chunk_sizes[q_h][index])/\
            1000000./future_bandwidth # this is MB/MB/s --> seconds

        rebuffer_time_l = max(download_time_l - start_buffer, 0)
        rebuffer_time_h = max(download_time_h - start_buffer, 0)

        Rebuf_p_l = rebuf_p * rebuffer_time_l
        Rebuf_p_h = rebuf_p * rebuffer_time_h

        # pdb.set_trace()

        ## Criterion for reducing the action space
        if last_q <= chunk_q_l: # alpha * (quality_low - quality_high) - beta (rebuff_low - rebuff_high) >= epsilon
            if quality_p * (chunk_q_l - chunk_q_h) - \
                rebuf_p * (Rebuf_p_l - Rebuf_p_h) >= EPSILON:
                break
        elif last_q >= chunk_q_h: # (alpha + s_p + s_n) * (quality_low - quality_high) - beta (rebuff_low - rebuff_high) >= epsilon
            if (quality_p + smooth_p + smooth_n) * (chunk_q_l - chunk_q_h) - \
                rebuf_p * (Rebuf_p_l - Rebuf_p_h) >= EPSILON:
                break 
        else:
            if (quality_p + smooth_p + smooth_n) * chunk_q_l - quality_p * chunk_q_h - \
                (smooth_p + smooth_n) * last_q - rebuf_p * (Rebuf_p_l - Rebuf_p_h) >= EPSILON:
                break 

    return output, future_bandwidth

def load_models(path, model_actor, model_critic):
    params_actor = torch.load(path + '/actor.pt')
    params_critic = torch.load(path + '/critic.pt')
    # model_actor.load_state_dict(params_actor)
    # model_critic.load_state_dict(params_critic)
    return model_actor.load_state_dict(params_actor), model_critic.load_state_dict(params_critic)

def save_models(logging, summary_dir, add_str, model_actor, vae_net, epoch,
                max_QoE, mean_value):
    save_path = summary_dir + '/' + add_str + '/models'
    if not os.path.exists(save_path): os.system('mkdir ' + save_path)
    
    save_flag = False
    if len(max_QoE) < 5: # save five models that perform best
        save_flag = True
        max_QoE[epoch] = mean_value
    elif mean_value > min(max_QoE.values()):
        min_idx = 0
        for key, value in max_QoE.items(): ## find the model that should be remove 
            if value == min(max_QoE.values()):
                min_idx = key if key > min_idx else min_idx

        actor_remove_path = summary_dir + '/' + add_str + '/' + 'models' + \
                        "/%s_%s_%d.model" %(str('policy'), add_str, int(min_idx))
        vae_remove_path = summary_dir + '/' + add_str + '/' + 'models' + \
                        "/%s_%s_%d.model" %(str('VAE'), add_str, int(min_idx))
        if os.path.exists(actor_remove_path): os.system('rm ' + actor_remove_path)
        # if os.path.exists(critic_save_path): os.system('rm ' + critic_save_path)
        if os.path.exists(vae_remove_path): os.system('rm ' + vae_remove_path)

        save_flag = True
        max_QoE.pop(min_idx)
        max_QoE[epoch] = mean_value
        
    if save_flag:
        logging.info("Model saved in file")
        # save models
        actor_save_path = summary_dir + '/' + add_str + '/' + 'models' + \
                            "/%s_%s_%d.model" %(str('policy'), add_str, int(epoch))
        # critic_save_path = summary_dir + '/' + add_str + "/%s_%s_%d.model" %(str('critic'), add_str, int(epoch))
        vae_save_path = summary_dir + '/' + add_str + '/' + 'models' + \
                            "/%s_%s_%d.model" %(str('VAE'), add_str, int(epoch))
        if os.path.exists(actor_save_path): os.system('rm ' + actor_save_path)
        # if os.path.exists(critic_save_path): os.system('rm ' + critic_save_path)
        if os.path.exists(vae_save_path): os.system('rm ' + vae_save_path)
        torch.save(model_actor.state_dict(), actor_save_path)
        # torch.save(model_critic.state_dict(), critic_save_path)
        torch.save(vae_net.state_dict(), vae_save_path)

def save_models_comyco(logging, summary_dir, add_str, model_actor, epoch,
                max_QoE, mean_value):
    save_path = summary_dir + '/' + add_str + '/models'
    if not os.path.exists(save_path): os.system('mkdir ' + save_path)
    
    save_flag = False
    if len(max_QoE) < 5: # save five models that perform best
        save_flag = True
        max_QoE[epoch] = mean_value
    elif mean_value > min(max_QoE.values()):
        min_idx = 0
        for key, value in max_QoE.items(): ## find the model that should be remove 
            if value == min(max_QoE.values()):
                min_idx = key if key > min_idx else min_idx

        actor_remove_path = summary_dir + '/' + add_str + '/' + 'models' + \
                        "/%s_%s_%d.model" %(str('policy'), add_str, int(min_idx))
        vae_remove_path = summary_dir + '/' + add_str + '/' + 'models' + \
                        "/%s_%s_%d.model" %(str('VAE'), add_str, int(min_idx))
        if os.path.exists(actor_remove_path): os.system('rm ' + actor_remove_path)
        # if os.path.exists(critic_save_path): os.system('rm ' + critic_save_path)
        if os.path.exists(vae_remove_path): os.system('rm ' + vae_remove_path)

        save_flag = True
        max_QoE.pop(min_idx)
        max_QoE[epoch] = mean_value
        
    if save_flag:
        logging.info("Model saved in file")
        # save models
        actor_save_path = summary_dir + '/' + add_str + '/' + 'models' + \
                            "/%s_%s_%d.model" %(str('policy'), add_str, int(epoch))

        if os.path.exists(actor_save_path): os.system('rm ' + actor_save_path)
        torch.save(model_actor.state_dict(), actor_save_path)

def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def clear_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        os.system('rm ' + folder_path +'/*')