class Config:
    # this is the model that does Faster-RCNN bounding box. Input should be RGB image
    model_weights = "/data2/SrikanthData/ResearchWork2/Latest_build/20221219-unet/Final_Models/Ensemble_Pipeline/pipeline_ensemble_ensemble/checkpoints/mobilenet-model-96-mAp-0.9600836177805456.pth"


    astrocyte_model_list = [
        {
            "architecture": "unet++",
            "encoder_name": "vgg16",
            "weight": "/data2/SrikanthData/ResearchWork2/Latest_build/20221219-unet/Final_Models/Ensemble_Pipeline/lightning_logs/version_0/checkpoints/epoch=158-step=2703.ckpt"
        },
        {
            "architecture": "manet",
            "encoder_name": "resnet152",
            "weight": "/data2/SrikanthData/ResearchWork2/Latest_build/20221219-unet/Final_Models/Ensemble_Pipeline/lightning_logs/version_1/checkpoints/epoch=147-step=2516.ckpt"
        },
        {
            "architecture": "unet++",
            "encoder_name": "resnet152",
            "weight": "/data2/SrikanthData/ResearchWork2/Latest_build/20221219-unet/Final_Models/Ensemble_Pipeline/lightning_logs/version_2/checkpoints/epoch=161-step=2754.ckpt"
        }
    ]

    nuclei_model_list = [
        {
            "architecture": "unet++",
            "encoder_name": "vgg16",
            "weight": "/data2/SrikanthData/ResearchWork2/Latest_build/20221219-unet/Final_Models/Ensemble_Pipeline/lightning_logs/version_3/checkpoints/epoch=56-step=1653.ckpt"
        },
        {
            "architecture": "manet",
            "encoder_name": "resnet152",
            "weight": "/data2/SrikanthData/ResearchWork2/Latest_build/20221219-unet/Final_Models/Ensemble_Pipeline/lightning_logs/version_4/checkpoints/epoch=174-step=5075.ckpt"
        },
        {
            "architecture": "unet++",
            "encoder_name": "resnet152",
            "weight": "/data2/SrikanthData/ResearchWork2/Latest_build/20221219-unet/Final_Models/Ensemble_Pipeline/lightning_logs/version_5/checkpoints/epoch=204-step=5945.ckpt"
        }
    ]

#     # this is the model that does only nuclues + nucleus border +  vs background segmentation
#     nuclei_model_list = [
# #         {
# #             "architecture": "unet++",
# #             "encoder_name": "vgg16"
# #             "weight": "/data2/SrikanthData/ResearchWork2/Latest_build/20221219-unet/Final_Models/Ensemble_Pipeline/lightning_logs/version_3/checkpoints/epoch=56-step=1653.ckpt"
# #         },
#         {
#             "architecture": "manet"
#             "encoder_name": "resnet152"
#             "weight": "/data2/SrikanthData/ResearchWork2/Latest_build/20221219-unet/Final_Models/Ensemble_Pipeline/lightning_logs/version_4/checkpoints/epoch=174-step=5075.ckpt"
#         },
#         {
#             "architecture": "unet++"
#             "encoder_name": "resnet152"
#             "weight": "/data2/SrikanthData/ResearchWork2/Latest_build/20221219-unet/Final_Models/Ensemble_Pipeline/lightning_logs/version_5/checkpoints/epoch=204-step=5945.ckpt"
#         }
#     ]



    # path to the image you want to test
    image_path = "/data2/SrikanthData/ResearchWork2/Latest_build/20221219-unet/Final_Models/Img_data/Data/astrocyte_coco/test_images/RGB/c2/RGB77.tif"

    gpu_id = '1'
    num_classes = 1 + 1
    data_root_dir = "/data2/SrikanthData/ResearchWork2/Latest_build/20221219-unet/Final_Models/Img_data/Data/astrocyte_coco"


test_cfg = Config()
