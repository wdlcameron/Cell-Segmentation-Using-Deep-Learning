from fastai.vision import *
from accuracy_functions import acc_segmentation, back_acc_segmentation, cell_acc_segmentation
from helper_functions import get_data 

def unet_trainer_program (root_path, export_name = 'Export.pkl', lr = 1e-4, 
                          size = 400, bs = 4, tfms = get_transforms(), data = None,
                          model = models.resnet34, output_cache = False,
                          reuse_old_model = False, old_model_name = 'Starter',
                          train_mode = True, wd = 1e-2, return_cache = False, penalty = 1):
    
    
    mask_path = root_path/"Mask_Norm" ; mask_path.mkdir(exist_ok = True)
    seg_path = root_path/"FullImage" ; #seg_path.mkdir(exist_ok = True)

    model_path = root_path/'Models' ; model_path.mkdir(exist_ok = True)
    figures_path = root_path/'Figures' ; figures_path.mkdir(exist_ok = True)


    
    #Padding Mode is an important parameter to check later!!  Mayeb set to zeros
    if data is None: data = get_data(root_path, tfms, size, bs)
    
    
    learn = unet_learner(data, model, wd = wd, metrics = [acc_segmentation, back_acc_segmentation, cell_acc_segmentation],
                         loss_func = CrossEntropyFlat(axis=1, weight = torch.FloatTensor([1,penalty]).cuda())
                       )
    
    
    #learn.save(model_path/'Starter')
    if reuse_old_model:  learn.load(old_model_name)
    
    cache = initialize_cache() #This will store information about the training, like the loss 
    
    if train_mode:
        learn.fit_one_cycle(5, max_lr = lr) #5
        cache = update_cache(learn, cache)
        learn.unfreeze()
        learn.fit_one_cycle(100, slice(lr/100, lr/2)) #100
        cache = update_cache(learn, cache)


        learn.export(model_path/export_name)


        #If you want to output the cache results
        if output_cache:
            output_cache_results(figures_path, cache)
    
    if return_cache: return learn, cache
    else: return learn

    
def initialize_cache():
    #Filler function for now
    return {}

def update_cache(learn, cache):    
    items = [('Loss', np.asarray(learn.recorder.losses)),
             ('Accuracy', np.asarray(learn.recorder.metrics))
            ]
    for name, values in items:
        if name in cache: cache[name] = np.concatenate((cache[name], values))
        else: cache[name] = values
    return cache
    
    

def output_cache_results(output_path, cache):
    loss_results = cache['Loss']
    accuracy_results = cache['Accuracy']
    
    
    for name, results in [("Loss Results", loss_results),("Accuracy Results",accuracy_results)]:
        plt.plot(results)
        plt.title(name)
        plt.xlabel("Epochs")  #Note, the loss outputs aren't for epochs...
        plt.savefig(output_path/f"{name}-cache.png")
        plt.close()
        
        np.savetxt(output_path/f"{name}-cache.txt", results)
    return None


#This function is now obsolete
def output_results(output_path, learn):
    loss_results = np.asarray(learn.recorder.losses)
    accuracy_results = np.asarray(learn.recorder.metrics)
    
    
    for name, results in [("Loss Results", loss_results),("Accuracy Results",accuracy_results)]:
        plt.plot(results)
        plt.title(name)
        plt.xlabel("Epochs")  #Note, the loss outputs aren't for epochs...
        plt.savefig(output_path/f"{name}.png")
        plt.close()
        
        np.savetxt(output_path/f"{name}.txt", results)
        
    return None