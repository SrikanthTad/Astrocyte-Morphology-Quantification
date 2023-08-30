1) To train the Segnmentation Ensemble using 'unet-ensemble-nuclei.ipynb' and 'unet-ensemble-astrocyte.ipynb'
    * Save the model_list from that notebook to the Faster-RCNN-ensemble/config/test-config.py
    * In that file, update the path to the weight of each model too
    * In that same file, update the path to the RCNN model (that does box prediction of astrocyte) and astrocyte segmentation

2) set the image use for prediction 

3) run the generate-d.py
