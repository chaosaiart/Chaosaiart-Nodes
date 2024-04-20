import cv2 
from tqdm import tqdm
import shutil

import torch

import os
import sys
import json
import hashlib
import traceback
import math
import re

import time
import glob
import random

 
from PIL import Image, ImageOps, ImageSequence, ImageEnhance
from PIL.PngImagePlugin import PngInfo
import numpy as np
import safetensors.torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))


import comfy.diffusers_load
import comfy.samplers
import comfy.sample
import comfy.sd
import comfy.utils
import comfy.controlnet

import comfy.clip_vision

import comfy.model_management
from comfy.cli_args import args

import importlib

import folder_paths
import latent_preview


 
# Hack: string type that is always equal in not equal comparisons
class AnyType_fill(str):
    def __ne__(self, __value: object) -> bool:
        return False

# Our any instance wants to be a wildcard string
anyType = AnyType_fill("*")

aSaveIMG = []  
aSaveLatent = []  
aSaveAny = []  
aSaveKsampler = []
MAX_RESOLUTION=8192

logActiv = {
    "lora_load" : False,
    "error": True,
    "Debugger": False 
} 

oCache_CKPTname_LoraName_Array_promtTxt = []

class anySave:
    def __init__(self, index, saveItem): 
        self.index = index
        self.saveItem = saveItem


class chaosaiart_higher: 
    def name(name_Type): 
        main_name = "🔶Chaosaiart"

        categories = {
            "main":         main_name,
            "checkpoint":   main_name+"/checkpoint",
            "image":        main_name+"/image",
            "prompt":       main_name+"/prompt",
            "ksampler":     main_name+"/ksampler",
            "cache":        main_name+"/cache",
            "restart":      main_name+"/restart",
            "logic":        main_name+"/logic",
            "controlnet":   main_name+"/controlnet",
            "lora":         main_name+"/lora",
            "video":        main_name+"/video",
            "switch":       main_name+"/switch",
            "animation":    main_name+"/animation",
        }

        out = categories.get(name_Type)
        if out == None:
            return categories.get("main")
        return out

    def log(Node,msg,Activ_status):
        if Activ_status:
            print(Node+": "+msg)

    def Debugger(Node, msg):
        if logActiv["Debugger"]:
            print("Debugger: "+Node)
            print(msg)

    def ErrorMSG(Node,msg):
        print(f"\033[91mERROR: {Node}: {msg}\033[0m")


   

    @classmethod
    def check_Checkpoint_Lora_Txt_caches(cls,index,sCKPT_name,aLora_Array,sPositiv_txt,sNegativ_txt):
        
        bCKPT_inCache   = False
        bLora_inCache   = False
        bPositiv_inCache = False
        bNegativ_inCache = False 

        iIndex = None

        if index is not None: 
            if 0 <= index < len(oCache_CKPTname_LoraName_Array_promtTxt):
                iIndex = index
                o = oCache_CKPTname_LoraName_Array_promtTxt[iIndex]
                if sCKPT_name == o.get("ckpt") and sCKPT_name is not None:
                    bCKPT_inCache = True
                    if aLora_Array == o.get("lora") and aLora_Array is not None:
                        bLora_inCache = True
                        if sPositiv_txt == o.get("positiv") and sPositiv_txt is not None: 
                            bPositiv_inCache = True
                        if sNegativ_txt == o.get("negativ") and sNegativ_txt is not None: 
                            bNegativ_inCache = True
                            
        if iIndex is None: 
            oCache_CKPTname_LoraName_Array_promtTxt.append({"ckpt": sCKPT_name, "lora": aLora_Array, "positiv":sPositiv_txt, "negativ":sNegativ_txt})
            iIndex = len(oCache_CKPTname_LoraName_Array_promtTxt) - 1
        else:
            oCache_CKPTname_LoraName_Array_promtTxt[iIndex] = {"ckpt":sCKPT_name,"lora":aLora_Array,"positiv":sPositiv_txt, "negativ":sNegativ_txt}

        return iIndex, bCKPT_inCache, bLora_inCache, bPositiv_inCache ,bNegativ_inCache
    
  
    def textClipEncode(clip,text):
        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return [[cond, {"pooled_output": pooled}]]
    
    """ #TODO: FIXME: NOTIZ:
    @classmethod    
    def CKPT_new_or_cache(cls,Checkpoint_name,Cached_CKPT_Name,Cached_CKPT):
        bNewCheckpoint = False
        if not Cached_CKPT_Name == Checkpoint_name or Cached_CKPT is None:: 
            Cached_CKPT = cls.checkpointLoader(Checkpoint_name)
            bNewCheckpoint = True
        return Cached_CKPT, Checkpoint_name, bNewCheckpoint
    """
    def checkpointLoader(ckpt_name):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))     
        return out[:3]
    
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def adjust_contrast(image, factor): 
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    def adjust_saturation(image, factor): 
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)

    def adjust_brightness(image, factor): 
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)
    
    def adjust_primary_colors(image, red_factor, green_factor, blue_factor): 
        bands = image.split()
        bands = [b.point(lambda x: x * red_factor) if i == 0 else
                 b.point(lambda x: x * green_factor) if i == 1 else
                 b.point(lambda x: x * blue_factor) for i, b in enumerate(bands)]
        return Image.merge(image.mode, bands)
    
    def ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
        latent_image = latent["samples"]
        if disable_noise:
            noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
        else:
            batch_inds = latent["batch_index"] if "batch_index" in latent else None
            noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        callback = latent_preview.prepare_callback(model, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                    denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                    force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
        out = latent.copy()
        out["samples"] = samples
        return (out, )
    
    @classmethod
    def reloader_x(cls, art, index, save, Input):
        
        global aSaveIMG
        global aSaveLatent
        global aSaveAny
        global aSaveKsampler
        
        aTemp = []
        if art == "img":
            aTemp = aSaveIMG
        elif art == "latent":
            aTemp = aSaveLatent
        elif art == "ksampler":
            aTemp = aSaveKsampler
        else:
            aTemp = aSaveAny

        if save:
            empty_Check =True
            for object in aTemp:
                if object.index == index:
                    empty_Check =False
                    object.saveItem = Input
                    break

            if empty_Check:
                aTemp.append(anySave(index,Input))
            
            if art == "img":
                aSaveIMG = aTemp
            elif art == "latent":
                aSaveLatent = aTemp
            else:
                aSaveAny = aTemp

        else:
            for object in aTemp:
                if object.index == index:
                    return (object.saveItem)
            cls.ErrorMSG(f"Chaosaiart:",f"{art}_Cache nr: {index} not exist") 
            raise ValueError(f"Chaosaiart: {art}_Cache nr: {index} not exist")

    def check_Text(txt):
        
        try: 
            if txt is None:
                return False
            if txt == "":
                return False
        
            str(txt) 
            return True
        except ZeroDivisionError:
            return False
    

        
    def replace_randomPart(text): 
        #new function.
        matches = re.findall(r'\{([^{}]*)\}', text)
         
        for match in matches: 
            options = match.split('|') 
            replacement = random.choice(options) 
            text = text.replace('{' + match + '}', replacement, 1)
         
        text = text.replace('|', '').replace('{', '').replace('}', '')
        
        return text

    @classmethod    
    def add_Prompt_txt__replace_randomPart(cls, txt, txt2):
        if cls.check_Text(txt2):
            if cls.check_Text(txt):
                return cls.replace_randomPart(txt) + "," + cls.replace_randomPart(txt2)
            return cls.replace_randomPart(txt2)
        
        if cls.check_Text(txt):
            return cls.replace_randomPart(txt)
            
        return ""     
    
    @classmethod
    def add_Prompt_txt_byMode(cls, txt, txt2, txt_after=True):
        if txt_after:
            return cls.add_Prompt_txt__replace_randomPart(txt, txt2)
        return cls.add_Prompt_txt__replace_randomPart(txt2, txt) 
    
    def make_array(a1,a2,a3,a4,a5,a6,a7,a8,a9):
        mArray = [a1,a2,a3,a4,a5,a6,a7,a8,a9]
        outArray = sorted(filter(None, mArray), key=lambda x: x[0], reverse=True)

        return outArray
      
    def get_Element_by_Frame(activ_frame,mArray):    

        for array in mArray:
            if isinstance(array, list): 
                if activ_frame >= int(array[0]):
                    return array

        return None            
     
    def round(Number):
        lower = math.floor(Number)
        higher = math.ceil(Number)
        
        distanc_lower = abs(Number - lower)
        distanc_higher = abs(Number - higher)
        
        if distanc_lower < distanc_higher:
            return lower
        else:
            return higher
        
    @classmethod    
    def video2frame(cls, VideoPath,output_dir,FPS_Mode):
        video_file = VideoPath

        video_file = video_file.replace('"', '')
        output_dir = output_dir.replace('"', '')


        info = ""
        if not os.path.isfile(video_file):
            info = 'chaosaiart_video2img: Path failed, no Video'
            print(info) 
            return info,
        
        _, video_ext = os.path.splitext(video_file)
        allowed_formats = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv']

        if video_ext.lower() not in allowed_formats:
            info = f'Video have to be: {", ".join(allowed_formats)}'
            print(info) 
            return info,
         

        # Basename für den neuen Ordner
        new_folder_basename = 'v_0001'

        # Den vollständigen Pfad des neuen Ordners erstellen
        new_folder_path = os.path.join(output_dir, new_folder_basename) 
        #new_folder_path = output_dir 

        # Überprüfen, ob der Ordner bereits vorhanden ist
        folder_number = 2
        while os.path.exists(new_folder_path): 
            new_folder_basename = f'v_{folder_number:04d}'
            new_folder_path = os.path.join(output_dir, new_folder_basename)
            #new_folder_path = f"{output_dir}_{folder_number:04d}"
            folder_number += 1
        
        os.makedirs(new_folder_path)
        output_folder_end = new_folder_path
 
        # Öffne das Video
        video = cv2.VideoCapture(video_file)

        # Zähle die Anzahl der Frames und die Framerate des Videos
        gesamt_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        fps = cls.round(fps)

        FPS_info = int(fps)

        if FPS_Mode == "Low FPS":  
            FPS_info = int(fps/2)
        if FPS_Mode == "Lower FPS": 
            FPS_info = int(fps/3)
        if FPS_Mode == "Ultra Low FPS":
            FPS_info = int(fps/4)
        
        FPS_info = cls.round(FPS_info) 

        # Erstelle die Textdatei mit den Frames pro Sekunde
        fps_textdatei = os.path.join(output_folder_end, 'FPS_Info.txt')
        with open(fps_textdatei, 'w') as f: 
            f.write(f'FPS: {FPS_info}\n') 

        print(f'FPS: {FPS_info}') 
        print(f'Process started, please wait...')  

        # Schleife über alle Frames im Video
        frame_num = 0
        bild_num = 0
        LowFPS_jump = 0 

        with tqdm(total=gesamt_frames) as pbar:
            while True:
                # Lese den aktuellen Frame
                _, frame = video.read()

                # Überprüfe, ob das Ende des Videos erreicht ist
                if frame is None:
                    break
                
                frame_num += 1

                # Aktualisiere den Fortschrittsbalken
                pbar.update(1)

                if bild_num > 0:
                    if FPS_Mode == "Low FPS": 
                        if LowFPS_jump == 1:
                            LowFPS_jump = 0
                        else:
                            LowFPS_jump += 1
                            continue
                    if FPS_Mode == "Lower FPS":
                        if LowFPS_jump == 2:
                            LowFPS_jump = 0
                        else:
                            LowFPS_jump += 1
                            continue
                    if FPS_Mode == "Ultra Low FPS":
                        if LowFPS_jump == 3:
                            LowFPS_jump = 0
                        else:
                            LowFPS_jump += 1
                            continue

                # Generiere den Bildnamen 
                IMG_name = f'{bild_num:08d}.jpg'
                bild_pfad = os.path.join(output_folder_end, IMG_name)

                # Speichere das Bild
                cv2.imwrite(bild_pfad, frame)
 
                #frame_num += 1
                bild_num += 1

        # Schließe das Video
        video.release()
         
        info  = f'Frames: "{bild_num}"\n'
        info += f'FPS: "{FPS_info}"\n'
        info += f'Folder: \n"{output_folder_end}"\n\n'
        info += f'FPS-Info.txt: \n"{fps_textdatei}"\n'

        return info
        
        #cv2.destroyAllWindows()    
    
      
    def lora_mainprompt_and_frame(mainPrompt_lora, frame_Lora):
        
        if not mainPrompt_lora == []:
            if not frame_Lora == []:
                return mainPrompt_lora + frame_Lora
            return mainPrompt_lora   
        
        if not frame_Lora == []:
            return frame_Lora
        
        return []
         

    def add_Lora(add_lora,loraType,lora_name,strength_model,strength_clip):
        #TODO: maybe exption, wenn lora name nit vorhanden.

        strength_model_float = min(max(strength_model, -20), 20) 
        strength_clip_float = min(max(strength_clip, -20), 20) 

        loraObjekt = {"lora_type":loraType,"lora_name":lora_name,"strength_model":strength_model_float,"strength_clip":strength_clip_float}

        if add_lora is None:
            return [loraObjekt]
        
        add_lora.append(loraObjekt)
        return add_lora
  
    #NOTIZ: Used Lora_Name, not Lora Path
    
    @classmethod   
    def Check_all_loras_in_cacheArray(cls,cacheArray, loraArray):

        try:
            if not cacheArray == []:
                if len(loraArray) == len(cacheArray):
                    for i in range(len(loraArray)):  

                        if cacheArray[i] is not None:
                            if hasattr(loraArray[i], 'lora_name'):
                                lora_name  = loraArray[i].lora_name
                                cache_lora = cacheArray[i]
                                
                                if cache_lora is not None:
                                    if not cache_lora[0] == lora_name:
                                        #Other Lora then Cache, Delete Cache
                                        return False
                                else:
                                    #Not Found in Cache, Delete Cache
                                    return False
                            else:
                                #TODO: Attribut without lora_Name : Bug, 
                                #better Skip then Delete ? Thing what happen if ... 
                                return False
                        else:
                            #Not Found in Cache, Delete Cache
                            return False
                else: 
                    #Not Same Amount of Lora, Delete Cache
                    return False
            else: 
                #No Cache
                return False
        except NameError:
            #Error in Cache Delete Cache 
            cls.ErrorMSG("Chaosaiart-Load Lora","Can't Use Lora out of Cache")
            return False
        
        return True
        
             
    @classmethod   
    def load_lora(cls,cache_lora, model, clip, lora_name, strength_model, strength_clip):
        model_lora = model
        clip_lora  = clip
        try: 
            if not (strength_model == 0 and strength_clip == 0): 

                loaded_lora = cache_lora

                lora_path = folder_paths.get_full_path("loras", lora_name)
                lora = None

                if loaded_lora is not None:
                    if loaded_lora[0] == lora_name:
                        lora = loaded_lora[1]

                if lora is None:
                    lora = comfy.utils.load_torch_file(lora_path, safe_load=True) 
                    loaded_lora = [lora_name, lora] #NOTIZ: lora_path -> lora_name

                model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)

            else: 
                cls.log("Chaosaiart-Load Lora",f"skip: {lora_name}, Strengt = 0",logActiv["lora_load"]) 
        except NameError: 
            cls.ErrorMSG("Chaosaiart-Load Lora",f"can't use the Lora - {lora_name}") 

        return (model_lora, clip_lora,loaded_lora)
     
     #TODO: FIXME: NOTIZ:
    #@classmethod   
    #def load_lora_by_Array_or_cache(cls,lora, model, clip, cache_loraArray):
        


    @classmethod   
    def load_lora_by_Array(cls,lora, model, clip, cache_loraArray):
        loraArray = lora
        model_lora = model
        clip_positive  = clip
        clip_negative  = clip
        newCache = []
        info = "Lora:"


        #try: 
            #loaded_lora
        if loraArray is None: 
            cls.Debugger("Chaosaiart-Load Lora: no Lora",loraArray)
            return model_lora, clip_positive, clip_negative, newCache

        if not type(loraArray) == list:
            cls.Debugger("Chaosaiart-Load Lora",loraArray)
            return model_lora, clip_positive, clip_negative, newCache
        

        cls.log("Chaosaiart-Load Lora","Loading Lora...",logActiv["lora_load"])
        cls.Debugger("Chaosaiart-Load Lora",loraArray)
        for i in range(len(loraArray)):  

            #if hasattr(loraArray[i], 'lora_name') and hasattr(loraArray[i], 'lora_type') and hasattr(loraArray[i], 'strength_model') and hasattr(loraArray[i], 'strength_clip'):
            if "lora_type" in loraArray[i] and "lora_name" in loraArray[i] and "strength_model" in loraArray[i] and "strength_clip" in loraArray[i]:
    
                clip_type       = loraArray[i]["lora_type"]
                lora_name       = loraArray[i]["lora_name"]
                strength_model  = loraArray[i]["strength_model"]
                strength_clip   = loraArray[i]["strength_clip"]
                 
                info += "\n" + cls.path2name(lora_name)

                if cache_loraArray is not None: 
                    if type(cache_loraArray) == list:
                        if i < len(cache_loraArray):
                            cache_lora = cache_loraArray[i]
                        else:
                            cache_lora = None
                    else:
                        cache_lora = None

                try: 
                    if clip_type == "negativ":
                        model_lora, clip_negative, cache_Element = cls.load_lora(cache_lora, model_lora, clip_negative, lora_name, strength_model, strength_clip)
                    else:
                        model_lora, clip_positive, cache_Element = cls.load_lora(cache_lora, model_lora, clip_positive, lora_name, strength_model, strength_clip)
                        

                except NameError:  
                    cls.ErrorMSG("Chaosaiart-Load Lora",f"Error, can't Load Lora: {lora_name}")
                    cls.log("Chaosaiart-Load Lora","Proof: 1. Lora Exist, 2. Same Base Model as Checkpoint",logActiv["lora_load"])
                    cache_Element = [lora_name, None]
            else: 
                try:
                    e_lora_name = loraArray[i]["lora_name"] if "lora_name" in loraArray[i] else "Unknow" 
                except NameError:  
                    e_lora_name = "Unknow"

                cls.ErrorMSG("Chaosaiart-Load Lora",f"Can't Load one Lora, Name: {e_lora_name}")
                cache_Element = [e_lora_name, None]
             
            newCache.append(cache_Element)
                 
        
        return model_lora, clip_positive, clip_negative, newCache, info
    
    def path2name(path):
        dateiname, _ = os.path.splitext(os.path.basename(path))    
        return dateiname
    
    @classmethod
    def lora_info(cls,loraArray): 
        info = "Lora:"
        for i in range(len(loraArray)):  
            if "lora_name" in loraArray[i]: 
                info += "\n" + cls.path2name(loraArray[i]["lora_name"])

        return info

    def emptyVideoSize(Mode,Size):
        sizeAttribut = {
            "360p":[640,360],
            "480p":[856,480],#4px heigher
            "HD":[1280,720],
            "Full_HD":[1920,1080],
            "QHD":[2560,1440],
            "4k":[3840,2160],
            "8k":[7680,4320]
        }
        temp_Size = sizeAttribut[Size] 
        temp2_Size = sizeAttribut[Size] 

        info = "" 
        HD_Check = 1080;

        if temp_Size[1] >= HD_Check: 
            temp_Size = sizeAttribut["Full_HD"]
            info += "Use Upscaler for this Size.\n"  

        #height
        batch_height = temp_Size[0]
        batch_width  = temp_Size[1]
        height = temp2_Size[0]
        width  = temp2_Size[1]

        if Mode == "widht":
            batch_height = temp_Size[1]
            batch_width  = temp_Size[0]
            height = temp2_Size[1]
            width  = temp2_Size[0]
        if Mode == "widht=height": 
            batch_height = temp_Size[1]
            batch_width  = temp_Size[1]
            height = temp2_Size[1]
            width  = temp2_Size[1]

        info = f"Output:\nlatent_size={batch_width}x{batch_height}x\nwidth:{width}\nheight:{height}"

        return  info,batch_height,batch_width,height,width 
    
    @classmethod
    def resize_image_pil(cls,imageIN,target_width, target_height):
        img = tensor2pil(imageIN) 
        resized_img = cls.resize_image("resize",img, target_width, target_height) 
        return pil2tensor(resized_img)

    @classmethod 
    def resize_image_fill_pil(cls,imageIN, target_width, target_height): 
        img = tensor2pil(imageIN)  
        resized_img = cls.resize_image("fill",img, target_width, target_height)   
        pil2tensor(resized_img)   

    @classmethod
    def resize_image_crop_pil(cls,imageIN, target_width, target_height):
        img = tensor2pil(imageIN)  
        resized_img = cls.resize_image("crop",img, target_width, target_height)   
        return pil2tensor(resized_img)  

    @classmethod 
    def resize_image(cls, resize_tpye,imageIN, target_width, target_height):   
        if resize_tpye == "resize":
            #return imageIN.resize((target_width, target_height), Image.ANTIALIAS) 
            return imageIN.resize((target_width, target_height), Image.Resampling.LANCZOS)

        elif resize_tpye == "crop" or resize_tpye == "fill": 
             
            img = imageIN

            width_ratio = target_width / img.width
            height_ratio = target_height / img.height
            
            if resize_tpye == "crop":
                scale_ratio = max(width_ratio, height_ratio)
            else: 
                scale_ratio = min(width_ratio, height_ratio)
            
            new_width = int(img.width * scale_ratio)
            new_height = int(img.height * scale_ratio)
            
            #resized_img = img.resize((new_width, new_height), Image.ANTIALIAS)
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            left = (new_width - target_width) / 2
            top = (new_height - target_height) / 2
            right = (new_width + target_width) / 2
            bottom = (new_height + target_height) / 2
            
            cropped_img = resized_img.crop((left, top, right, bottom))
            
            return cropped_img 
        else:
            cls.ErrorMSG("Chaosaiart_Resize Img:","No Resize Type please do a issue request include your workflow")
   
    @classmethod 
    def check_seed(cls,error_seed_name,seed):  
        try:   
            return int(seed)  
        except Exception as e:
            cls.ErrorMSG(cls.name("main"),f"{error_seed_name} : Unknow type of Seed, {error_seed_name} = new Seed")
            return -1

    @classmethod 
    def seed(cls,seed): 
        seed = cls.check_seed("seed",seed)
        if seed <= 0: 
            return random.randint(1, 0xfffffffffff) 
        return seed 
    
    @classmethod 
    def txt2video_SEED_cachSEED(cls,seed,cach_seed):  
        seed = cls.check_seed("seed",seed)
        cach_seed = cls.check_seed("cache_seed",cach_seed) 
        if seed <= 0:
            if cach_seed <= 0:
                new_seed = cls.seed(-1) 
                return new_seed,new_seed
            return cach_seed,cach_seed
        return seed, seed

   

 
class chaosaiart_CheckpointPrompt2:
    def __init__(self):

        self.Cache_index    = None
        self.Cache_CKPT     = None
        self.Cache_Lora     = None
        self.Cache_pPrompt  = None
        self.Cache_nPrompt  = None 

        self.lora_load_cache = []

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {  
                "Checkpoint": (folder_paths.get_filename_list("checkpoints"), ), 
                "Positiv": ("STRING", {"multiline": True}),
                "Negativ": ("STRING", {"multiline": True}),
            },
            "optional": {
                "add_positiv_txt": ("STRING", {"multiline": True, "forceInput": True}),
                "add_negativ_txt": ("STRING", {"multiline": True, "forceInput": True}),
                "add_lora":("LORA",),
            },
            
        }
    RETURN_TYPES = ("STRING","MODEL", "CONDITIONING", "CONDITIONING", "VAE",)
    RETURN_NAMES = ("Info","MODEL","POSITIV","NEGATIV","VAE",) 
    FUNCTION = "node"

    CATEGORY = chaosaiart_higher.name("checkpoint")
  
    def node(self, Checkpoint, Positiv="",Negativ="",add_lora=[],add_positiv_txt = "",add_negativ_txt=""):
        
    
        ckpt_name = Checkpoint 
        sPositiv = chaosaiart_higher.add_Prompt_txt__replace_randomPart(add_positiv_txt,Positiv)
        sNegativ = chaosaiart_higher.add_Prompt_txt__replace_randomPart(add_negativ_txt,Negativ)  
        alora = add_lora
        
        self.Cache_index, bCKPT_inCache, bLora_inCache, bPositiv_inCache ,bNegativ_inCache \
        = chaosaiart_higher.check_Checkpoint_Lora_Txt_caches(self.Cache_index,ckpt_name,alora,sPositiv,sNegativ)

        if not bCKPT_inCache:
            self.Cache_CKPT = chaosaiart_higher.checkpointLoader(ckpt_name) 
        MODEL, CLIP, VAE = self.Cache_CKPT    

        if not ( bCKPT_inCache and bLora_inCache):    
            self.Cache_Lora = chaosaiart_higher.load_lora_by_Array(alora, MODEL, CLIP, self.lora_load_cache)
        MODEL, positiv_clip, negativ_clip, self.lora_load_cache, lora_Info = self.Cache_Lora

        if not ( bCKPT_inCache and bLora_inCache and bPositiv_inCache):   
            self.Cache_pPrompt = chaosaiart_higher.textClipEncode(positiv_clip, sPositiv) 
        PositivOut = self.Cache_pPrompt
        
        if not ( bCKPT_inCache and bLora_inCache and bNegativ_inCache):
            self.Cache_nPrompt = chaosaiart_higher.textClipEncode(negativ_clip, sNegativ) 
        NegativOut = self.Cache_nPrompt  

        info  = "checkpoint: "+chaosaiart_higher.path2name(Checkpoint)+"\n" 
        info += f"Positiv:\n{sPositiv}\n"
        info += f"Negativ:\n{sNegativ}\n" 
        info += lora_Info
        info += "\n\nCheckpoint: " 
        info += "No Change " if bCKPT_inCache else "Changed"  
        info += "\n\nLora: " 
        info += "No Change " if bLora_inCache else "Changed"  
        info += "\n\nPrompt Positiv: " 
        info += "No Change " if bPositiv_inCache else "Changed"  
        info += "\n\nPrompt Negativ: " 
        info += "No Change " if bNegativ_inCache else "Changed"  
        
        
        return (info,MODEL,PositivOut,NegativOut,VAE,) 
 

class chaosaiart_EmptyLatentImage:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "Mode":(["Widescreen / 16:9","Portrait (Smartphone) / 9:16","Widht = Height" ],),
                "Size":(["360p","480p","HD","Full HD (Check Info)","QHD (No LATENT, Check Info)","4k (No LATENT, Check Info)","8k (Latent HD, Check Info)",],)
            }
        }
    RETURN_TYPES = ("LATENT","INT","INT","STRING",)
    RETURN_NAMES = ("LATENT","WIDTH","HEIGHT","Info",)
    FUNCTION = "node"


    CATEGORY = chaosaiart_higher.name("image")
 
    def node(self, Mode, Size):
        
        sizeMode = {
            "360p":"360p",
            "480p":"480p",#4px heigher
            "HD":"HD",
            "Full HD (Check Info)":"Full_HD",
            "QHD (No LATENT, Check Info)":"QHD",
            "4k (No LATENT, Check Info)":"4k",
            "8k (Latent HD, Check Info)":"8k"
        }
        screenMode = {
            "Widescreen / 16:9":"widht",
            "Portrait (Smartphone) / 9:16":"height",
            "Widht = Height":"widht=height" 
        } 

        info,batch_height,batch_width,height,width = chaosaiart_higher.emptyVideoSize(screenMode[Mode],sizeMode[Size])
   
        batch_size = 1
        latent = torch.zeros([batch_size, 4, batch_height // 8, batch_width // 8], device=self.device)
        return ({"samples":latent},width,height,info,)
          
 

class chaosaiart_CheckpointPrompt:
    def __init__(self):

        self.Cache_index    = None
        self.Cache_CKPT     = None
        self.Cache_Lora     = None
        self.Cache_pPrompt  = None
        self.Cache_nPrompt  = None  
 
        self.lora_load_cache = []

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {  
                "Checkpoint": (folder_paths.get_filename_list("checkpoints"), ),   
                "positiv_txt": ("STRING", {"multiline": True, "forceInput": True}),
                "negativ_txt": ("STRING", {"multiline": True, "forceInput": True}),
            }, 
            "optional":{
                "lora":("LORA",), 
            }
        }
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "VAE",)
    RETURN_NAMES = ("MODEL","POSITIV","NEGATIV","VAE",) 
    FUNCTION = "node"

    CATEGORY = chaosaiart_higher.name("checkpoint")

    def node(self, Checkpoint, positiv_txt="",negativ_txt="",lora=[]):
         
        ckpt_name = Checkpoint 
        sPositiv = positiv_txt
        sNegativ = negativ_txt
        aLora = lora

        self.Cache_index, bCKPT_inCache, bLora_inCache, bPositiv_inCache ,bNegativ_inCache \
        = chaosaiart_higher.check_Checkpoint_Lora_Txt_caches(self.Cache_index,ckpt_name,aLora,sPositiv,sNegativ)

        if not bCKPT_inCache:
            self.Cache_CKPT = chaosaiart_higher.checkpointLoader(ckpt_name) 
        MODEL, CLIP, VAE = self.Cache_CKPT   
  
        if not ( bCKPT_inCache and bLora_inCache):    
            self.Cache_Lora = chaosaiart_higher.load_lora_by_Array(aLora, MODEL, CLIP, self.lora_load_cache)
        MODEL, positiv_clip, negativ_clip, self.lora_load_cache, lora_Info = self.Cache_Lora

        if not ( bCKPT_inCache and bLora_inCache and bPositiv_inCache):   
            self.Cache_pPrompt = chaosaiart_higher.textClipEncode(positiv_clip, sPositiv) 
        PositivOut = self.Cache_pPrompt
        
        if not ( bCKPT_inCache and bLora_inCache and bNegativ_inCache):
            self.Cache_nPrompt = chaosaiart_higher.textClipEncode(negativ_clip, sNegativ) 
        NegativOut = self.Cache_nPrompt   
       

        return (MODEL,PositivOut,NegativOut,VAE,) 
  
"""
class chaosaiart_Style_Node:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {  
                "Model":(["SDXL","SD1_5"],),
                "Quality": (["Masterpiece",],),
                "Light": (["none",],),
                "Technique": (["none",],),
                "Color": (["none",],),
                "Mood": (["none",],), 
                "Subjekt": (["none",],), 
                "Style": (["none",],), 
            }, 
            "optional":{
                "add_prompt_txt":("STRING", {"multiline": True, "forceInput": True}), 
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt_txt",) 
    FUNCTION = "node"

    CATEGORY = chaosaiart_higher.name("prompt")

    def node(self,Model,Quality,Light,Technique,Color,Mood,Subjekt,Style): 

        return ("masterpiece,best quality, highres",)

class chaosaiart_Style_Node:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {  
                "Model":(["SDXL","SD1_5"],),
                "Quality": (["Masterpiece",],),
                "Light": (["none",],),
                "Technique": (["none",],),
                "Color": (["none",],),
                "Mood": (["none",],), 
                "Subjekt": (["none",],), 
                "Style": (["none",],), 
            }, 
            "optional":{
                "add_prompt_txt":("STRING", {"multiline": True, "forceInput": True}), 
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt_txt",) 
    FUNCTION = "node"

    CATEGORY = chaosaiart_higher.name("prompt")

    def node(self,Model,Quality,Light,Technique,Color,Mood,Subjekt,Style): 

        return ("masterpiece,best quality, highres",)

"""
    
class chaosaiart_CheckpointPrompt_Frame:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {  
                "Start_Frame":("INT",{"default": 1, "min": 1, "max": 1000000000, "step": 1}),
                "Checkpoint": (folder_paths.get_filename_list("checkpoints"), ),
                "Positiv": ("STRING", {"multiline": True}),
                "Negativ": ("STRING", {"multiline": True}),
            }, 
            "optional": {
                "lora":("LORA",),
            }
            
        }
    RETURN_TYPES = ("CKPT_PROMPT",)
    RETURN_NAMES = ("CKPT_PROMPT",) 
    FUNCTION = "node"

    CATEGORY = chaosaiart_higher.name("checkpoint")

    def node(self, Start_Frame,Checkpoint,Positiv="",Negativ="",lora=[]):  
        return ([Start_Frame,Checkpoint,Positiv,Negativ,lora],)
                
 
class chaosaiart_CheckpointPrompt_FrameMixer:
    def __init__(self):

        self.Cache_index    = None
        self.Cache_CKPT     = None
        self.Cache_Lora     = None
        self.Cache_pPrompt  = None
        self.Cache_nPrompt  = None 

        self.lora_load_cache = [] 

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {  
                "activ_frame": ("ACTIV_FRAME", ),
                "main_prompt": ("MAIN_PROMPT",),
                "ckpt_prompt_1":("CKPT_PROMPT",),  
            },
            "optional": {
                "ckpt_prompt_2":("CKPT_PROMPT",), 
                "ckpt_prompt_3":("CKPT_PROMPT",), 
                "ckpt_prompt_4":("CKPT_PROMPT",), 
                "ckpt_prompt_5":("CKPT_PROMPT",), 
                "ckpt_prompt_6":("CKPT_PROMPT",), 
                "ckpt_prompt_7":("CKPT_PROMPT",), 
                "ckpt_prompt_8":("CKPT_PROMPT",), 
                "ckpt_prompt_9":("CKPT_PROMPT",), 
            },
        }
    #RETURN_TYPES = ("STRING","MODEL", "CONDITIONING", "CONDITIONING", "VAE","CKPT_PROMPT")
    #RETURN_NAMES = ("Info","MODEL","POSITIV","NEGATIV","VAE","CKPT_PROMPT")
    RETURN_TYPES = ("STRING","MODEL", "CONDITIONING", "CONDITIONING", "VAE",)
    RETURN_NAMES = ("Info","MODEL","POSITIV","NEGATIV","VAE",)
    FUNCTION = "node"

    CATEGORY = chaosaiart_higher.name("checkpoint")

    def node(self, activ_frame,main_prompt,
              ckpt_prompt_1,
              ckpt_prompt_2=None,
              ckpt_prompt_3=None,
              ckpt_prompt_4=None,
              ckpt_prompt_5=None,
              ckpt_prompt_6=None,
              ckpt_prompt_7=None,
              ckpt_prompt_8=None,
              ckpt_prompt_9=None):
         

        mArray = chaosaiart_higher.make_array(
            ckpt_prompt_1,
            ckpt_prompt_2,
            ckpt_prompt_3,
            ckpt_prompt_4,
            ckpt_prompt_5,
            ckpt_prompt_6,
            ckpt_prompt_7,
            ckpt_prompt_8,
            ckpt_prompt_9
        )

        if activ_frame < 1:
            activ_frame = 1
 
        activ_checkpoint_prompt_frame = chaosaiart_higher.get_Element_by_Frame(activ_frame,mArray) 

        if activ_checkpoint_prompt_frame is None:
            print("Chaosaiart - CheckpointPrompt_FrameMixer: no checkpoint_prompt_frame with this Activ_Frame. checkpoint_prompt_frame1 will be Used")
            activ_checkpoint_prompt_frame = ckpt_prompt_1

        main_positiv = main_prompt[0]
        main_negativ = main_prompt[1]
        main_lora    = main_prompt[2]

        ckpt_name   = activ_checkpoint_prompt_frame[1]   
        Positiv     = activ_checkpoint_prompt_frame[2]
        Negativ     = activ_checkpoint_prompt_frame[3]
        frame_lora  = activ_checkpoint_prompt_frame[4]
 
        sPositiv = chaosaiart_higher.add_Prompt_txt__replace_randomPart(main_positiv,Positiv)
        sNegativ = chaosaiart_higher.add_Prompt_txt__replace_randomPart(Negativ,main_negativ)  
          
        alora = chaosaiart_higher.lora_mainprompt_and_frame(main_lora,frame_lora)


        self.Cache_index, bCKPT_inCache, bLora_inCache, bPositiv_inCache ,bNegativ_inCache \
        = chaosaiart_higher.check_Checkpoint_Lora_Txt_caches(self.Cache_index,ckpt_name,alora,sPositiv,sNegativ)

        if not bCKPT_inCache:
            self.Cache_CKPT = chaosaiart_higher.checkpointLoader(ckpt_name) 
        MODEL, CLIP, VAE = self.Cache_CKPT    

        if not ( bCKPT_inCache and bLora_inCache):    
            self.Cache_Lora = chaosaiart_higher.load_lora_by_Array(alora, MODEL, CLIP, self.lora_load_cache)
        MODEL, positiv_clip, negativ_clip, self.lora_load_cache, lora_Info = self.Cache_Lora
 
        if not ( bCKPT_inCache and bLora_inCache and bPositiv_inCache):   
            self.Cache_pPrompt = chaosaiart_higher.textClipEncode(positiv_clip, sPositiv) 
        PositivOut = self.Cache_pPrompt
        
        if not ( bCKPT_inCache and bLora_inCache and bNegativ_inCache):
            self.Cache_nPrompt = chaosaiart_higher.textClipEncode(negativ_clip, sNegativ) 
        NegativOut = self.Cache_nPrompt  
  
        info  = f"Frame:{activ_frame}\nCheckpoint: {activ_checkpoint_prompt_frame[1]}\nPostiv:\n {sPositiv}\nNegativ:\n {sNegativ}\n" 
        
        info += lora_Info 
        return (info,MODEL,PositivOut,NegativOut,VAE,) 
                
    
 
        
 
"""  
class chaosaiart_KSampler: 
    @classmethod
    def INPUT_TYPES(s):
        return {    
            "required":{
                    "model": ("MODEL",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "vae": ("VAE", ),
                    "latent_image": ("LATENT", ), 
                    "denoise": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
                }, 
                "optional":{ 
                    "denoise_Override": ("FLOAT",{"forceInput": True}),  
                }
            }

    #RETURN_TYPES = ("LATENT",)
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "node"

    CATEGORY = chaosaiart_higher.name("ksampler")

    def node(self, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise, denoise_Override=None):
        denoise = denoise if denoise_Override is None else denoise_Override 
        samples = chaosaiart_higher.ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)
        image = vae.decode(samples[0]["samples"]) 
        return (image,) 
  """
    


class chaosaiart_KSampler1: #txt2img
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {    
            "required":{
                    "model": ("MODEL",),
                    "Image_width": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                    "Image_height": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}), 
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "vae": ("VAE", ),  
                },
            }

    RETURN_TYPES = ("IMAGE",) 
    RETURN_NAMES = ("IMAGE",) 
    FUNCTION = "node"

    CATEGORY = chaosaiart_higher.name("ksampler")

    def node(self, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, Image_width,Image_height):

        Image_batch_size = 1
        denoise = 1  
        latent = torch.zeros([Image_batch_size, 4, Image_height // 8, Image_width // 8], device=self.device)
        latent_image = {"samples":latent} 
     

        samples = chaosaiart_higher.ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)
        image = vae.decode(samples[0]["samples"])
        return (image,samples[0],)
   

class chaosaiart_KSampler2: #img2img
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {    
            "required":{
                    "model": ("MODEL",),
                    "Image_Mode":(["Widht = Height","Widescreen / 16:9","Portrait (Smartphone) / 9:16"],),
                    "Image_Size":(["360p","480p","HD","Full HD",],),
                    "Img2img_input_Size":(["resize","crop","override"],), 
                    "denoise": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "vae": ("VAE", ), 
                    "image": ("IMAGE", ), 
                }, 
                "optional":{ 
                    "denoise_Override": ("FLOAT",{"forceInput": True}),   
                }
            }

    RETURN_TYPES = ("IMAGE",) 
    RETURN_NAMES = ("IMAGE",) 
    FUNCTION = "node"

    CATEGORY = chaosaiart_higher.name("ksampler")


 
    @staticmethod
    def vae_encode_crop_pixels(pixels):
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
        return pixels
 
    def node(self, model,Image_Mode,Image_Size,Img2img_input_Size, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, image,denoise,denoise_Override=None):
        
        sizeMode = { "360p":"360p", "480p":"480p", "HD":"HD", "Full HD":"Full_HD" }
        screenMode = { "Widescreen / 16:9":"widht", "Portrait (Smartphone) / 9:16":"height", "Widht = Height":"widht=height" } 
        infoSize, batch_height, batch_width, height, width = chaosaiart_higher.emptyVideoSize(screenMode[Image_Mode],sizeMode[Image_Size])   
        
        if Img2img_input_Size == "resize":
            image = chaosaiart_higher.resize_image_pil(image,batch_width,batch_height)
            
        if Img2img_input_Size == "crop": 
            image = chaosaiart_higher.resize_image_crop_pil(image,batch_width,batch_height)
                 
        denoise = denoise if denoise_Override is None else denoise_Override 

        pixels = image
        pixels = self.vae_encode_crop_pixels(pixels)
        t = vae.encode(pixels[:,:,:,:3])
        latent_image = {"samples":t} 

        samples = chaosaiart_higher.ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)
        image = vae.decode(samples[0]["samples"])
        return (image,samples[0],)

class chaosaiart_KSampler3: 
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {    
            "required":{
                    "model": ("MODEL",),
                    "Image_Mode":(["Widht = Height","Widescreen / 16:9","Portrait (Smartphone) / 9:16"],),
                    "Image_Size":(["360p","480p","HD","Full HD",],),
                    "Img2img_input_Size":(["resize","crop","override"],),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "vae": ("VAE", ),
                    "denoise": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
                },
                "optional":{ 
                    "denoise_Override": ("FLOAT",{"forceInput": True}),  
                    "latent_Override": ("LATENT", ), 
                    "latent_by_Image_Override": ("IMAGE", ), 
                }
            }

    RETURN_TYPES = ("IMAGE","LATENT",) 
    RETURN_NAMES = ("IMAGE","SAMPLES",) 
    FUNCTION = "node"

    CATEGORY = chaosaiart_higher.name("ksampler")
    
    @staticmethod
    def vae_encode_crop_pixels(pixels):
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
        return pixels
     

    def node(self, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise,Image_Mode,Image_Size,Img2img_input_Size,latent_Override=None,latent_by_Image_Override=None,denoise_Override=None):
        
        denoise = denoise if denoise_Override is None else denoise_Override 

        sizeMode = { "360p":"360p", "480p":"480p", "HD":"HD", "Full HD":"Full_HD" }
        screenMode = { "Widescreen / 16:9":"widht", "Portrait (Smartphone) / 9:16":"height", "Widht = Height":"widht=height" } 
        infoSize, batch_height, batch_width, height, width = chaosaiart_higher.emptyVideoSize(screenMode[Image_Mode],sizeMode[Image_Size])   
        

        
        if latent_by_Image_Override is None: 
            if latent_Override is None:
                batch_size = 1
                latent = torch.zeros([batch_size, 4, batch_height // 8, batch_width // 8], device=self.device)
                latent_image = {"samples":latent}
                if not denoise==1:
                    print("chaosaiart_KSampler2: set Denoising to 1")
                denoise = 1 
            else:
                #NOTIZ: No Resize. 
                latent_image = latent_Override

        else:
            if Img2img_input_Size == "resize":
                latent_by_Image_Override = chaosaiart_higher.resize_image_pil(latent_by_Image_Override,batch_width,batch_height)
                
            if Img2img_input_Size == "crop": 
                latent_by_Image_Override = chaosaiart_higher.resize_image_crop_pil(latent_by_Image_Override,batch_width,batch_height)
                
            pixels = latent_by_Image_Override
            pixels = self.vae_encode_crop_pixels(pixels)
            t = vae.encode(pixels[:,:,:,:3])
            latent_image = {"samples":t} 

        samples = chaosaiart_higher.ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)
        image = vae.decode(samples[0]["samples"])
        return (image,samples[0],)


class chaosaiart_KSampler4:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                        "model": ("MODEL",),
                        "Image_Mode":(["Widht = Height","Widescreen / 16:9","Portrait (Smartphone) / 9:16"],),
                        "Image_Size":(["360p","480p","HD","Full HD",],),
                        "Img2img_input_Size":(["resize","crop","override"],),
                        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                        "steps": ("INT", {"default": 25, "min": 1, "max": 10000}),
                        "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                        "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                        "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                        "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                        "positive": ("CONDITIONING", ),
                        "negative": ("CONDITIONING", ),
                        "vae": ("VAE", ),
                     },
                     "optional":{  
                        "latent_Override": ("LATENT", ),  
                        "latent_by_Image_Override": ("IMAGE", ),  
                        "start_at_step_Override": ("INT", {"forceInput": True} ),  
                    }
                }

    RETURN_TYPES = ("IMAGE","LATENT",) 
    RETURN_NAMES = ("IMAGE","SAMPLES",) 
    FUNCTION = "node"

    CATEGORY = chaosaiart_higher.name("ksampler")

    @staticmethod
    def vae_encode_crop_pixels(pixels):
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
        return pixels
    
    def node(self, 
             model, seed, steps, cfg, sampler_name, scheduler, positive, negative, start_at_step, 
             vae,Image_Mode,Image_Size,Img2img_input_Size,
             latent_Override = None, latent_by_Image_Override = None,start_at_step_Override = None,
             denoise=1.0):
        
        sizeMode = { "360p":"360p", "480p":"480p", "HD":"HD", "Full HD":"Full_HD" }
        screenMode = { "Widescreen / 16:9":"widht", "Portrait (Smartphone) / 9:16":"height", "Widht = Height":"widht=height" } 
        infoSize, batch_height, batch_width, height, width = chaosaiart_higher.emptyVideoSize(screenMode[Image_Mode],sizeMode[Image_Size])   
                

        return_with_leftover_noise = "disable" 
        add_noise = "enable"

        start_at_step = start_at_step if start_at_step_Override is None else start_at_step_Override
        end_at_step = steps

        if latent_by_Image_Override is None: 
            if latent_Override is None:
                batch_size = 1
                latent = torch.zeros([batch_size, 4, batch_height // 8, batch_width // 8], device=self.device)
                latent_image = {"samples":latent}
                start_at_step = 0
            else: 
                #NOTIZ: No Resize. 
                latent_image = latent_Override
        else:
            if Img2img_input_Size == "resize":
                latent_by_Image_Override = chaosaiart_higher.resize_image_pil(latent_by_Image_Override,batch_width,batch_height)
                
            if Img2img_input_Size == "crop": 
                latent_by_Image_Override = chaosaiart_higher.resize_image_crop_pil(latent_by_Image_Override,batch_width,batch_height)
        
            pixels = latent_by_Image_Override
            pixels = self.vae_encode_crop_pixels(pixels)
            t = vae.encode(pixels[:,:,:,:3])
            latent_image = {"samples":t} 

 

        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True 

        samples =  chaosaiart_higher.ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)
        image = vae.decode(samples[0]["samples"])
        return (image,samples[0],)
    
class chaosaiart_KSampler_expert_0:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()
        self.cache_latent = None
        self.last_activ_frame = 0
        self.started = False
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    { 
                        "model": ("MODEL",),     
                        "Image_Mode":(["Widht = Height","Widescreen / 16:9","Portrait (Smartphone) / 9:16"],),
                        "Image_Size":(["360p","480p","HD","Full HD",],),
                        "Img2img_input_Size":(["resize","crop","override"],),
                        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1}), 
                        "steps": ("INT", {"default": 25, "min": 0, "max": 10000, "step": 1}),
                        "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                        "end_at_step": ("INT", {"default": 25, "min": 0, "max": 10000, "step": 1}),
                        "denoise":("FLOAT", {"default": 1, "min": 0, "max": 1}),
                        "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                        "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                        "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                        "positive": ("CONDITIONING", ),
                        "negative": ("CONDITIONING", ), 
                        "vae": ("VAE", ),
                        "use_CacheImg":(["Enable","Disable"],),
                        "add_noise": (["enable", "disable"], ),
                        "return_with_leftover_noise": (["disable", "enable"], ),
                     },
                     "optional":{  
                        "activ_frame":("ACTIV_FRAME",),
                        "Latent_by_image_Override":("IMAGE",),  
                        "Latent_Override":("LATENT",),  
                        "start_at_step_Override": ("INT", {"forceInput": True} ),  
                    }
                }

    RETURN_TYPES = ("STRING","LATENT",) 
    RETURN_NAMES = ("Info","SAMPLER",) 
    FUNCTION = "node"

    CATEGORY = chaosaiart_higher.name("animation")

    @staticmethod
    def vae_encode_crop_pixels(pixels):
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
        return pixels
    
    def node(self, Image_Size,Image_Mode, Img2img_input_Size,
             use_CacheImg,return_with_leftover_noise, add_noise,
             model, seed, steps, cfg, sampler_name, scheduler, positive, negative, start_at_step,end_at_step, 
             vae, activ_frame = 0, 
             start_at_step_Override = None,
             denoise=1.0, Latent_by_image_Override=None, Latent_Override=None):
         
        start_at_step = start_at_step if start_at_step_Override is None else start_at_step_Override 
     
       
        sizeMode = { "360p":"360p", "480p":"480p", "HD":"HD", "Full HD":"Full_HD" }
        screenMode = { "Widescreen / 16:9":"widht", "Portrait (Smartphone) / 9:16":"height", "Widht = Height":"widht=height" } 
        infoSize, batch_height, batch_width, height, width = chaosaiart_higher.emptyVideoSize(screenMode[Image_Mode],sizeMode[Image_Size])   
          
        info = ""

        if (self.last_activ_frame >= activ_frame and activ_frame > 0):
            self.started = False
            
        latent = None
        if self.started and use_CacheImg == "Enable" and self.cache_latent is not None:
            info = "Use Cache Image\n"
            latent = self.cache_latent
        else:    
            if Latent_by_image_Override is not None:   
                if Img2img_input_Size == "resize":
                    Latent_by_image_Override = chaosaiart_higher.resize_image_pil(Latent_by_image_Override,batch_width,batch_height)
                    
                if Img2img_input_Size == "crop": 
                    Latent_by_image_Override = chaosaiart_higher.resize_image_crop_pil(Latent_by_image_Override,batch_width,batch_height)
             
                pixels = self.vae_encode_crop_pixels(Latent_by_image_Override)
                latent = vae.encode(pixels[:,:,:,:3])
                info = "Use Latent_by_image_Override\n"
            elif Latent_Override is not None:
                info = "Use Latent_Override\n"
                if not self.started:
                    info += "!! Denoise =1 !!\n"
                    denoise = 1
                latent = Latent_Override["samples"] 

        if latent is None: 
            start_at_step = 0 
            denoise = 1
            batch_size = 1   
            latent = torch.zeros([batch_size, 4, batch_height // 8, batch_width // 8], device=self.device)
            info = "Use new Empty Image\n" + infoSize

        latent_image = {"samples":latent} 
        
        
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True 

        samples =  chaosaiart_higher.ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)
        image = vae.decode(samples[0]["samples"]) 
        self.cache_latent = samples[0]["samples"]
        self.last_activ_frame = activ_frame
        self.started = True
        
        info = f"Frame: {activ_frame}\n" + info +f"\nseed: {seed}"
        return (info,samples[0]) 
    
  
     
class chaosaiart_KSampler_a1: #img2video
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()
        self.cache_latent = None
        self.last_activ_frame = 0
        self.started = False
        self.cache_seed = -1
        self.cache_seed_2 = -1
        self.img2video_mode = True
        #self.img2img_Size, self.Image_Size, self.Image_Mode =  None, None, None


    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    { 
                        "activ_frame":("ACTIV_FRAME",),
                        "model": ("MODEL",),     
                        "Image_Mode":(["Widht = Height","Widescreen / 16:9","Portrait (Smartphone) / 9:16"],),
                        "Image_Size":(["360p","480p","HD","Full HD",],),
                        "Img2img_input_Size":(["crop","override","resize"],),
                        "seed_start":("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff, "step": 1}),
                        "seed_mode":(["fixed - 0.50 - SD1.5",
                                      "increment",
                                      "fixed - 0.45", 
                                      "fixed - 0.48", 
                                      "fixed - 0.52", 
                                      "fixed - 0.55", 
                                      ],), 
                        "steps": ("INT", {"default": 25, "min": 0, "max": 10000, "step": 1}),
                        "denoise":("FLOAT", {"default": 0.7, "min": 0, "max": 1, "step":0.01}),
                        "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                        "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                        "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                        "positive": ("CONDITIONING", ),
                        "negative": ("CONDITIONING", ), 
                        "vae": ("VAE",),
                    },
                     "optional":{  
                        "start_Image":("IMAGE",),  
                        "zoom_frame":("ZOOM_FRAME",),  
                    }
                }

    RETURN_TYPES = ("STRING","IMAGE","LATENT",) 
    RETURN_NAMES = ("Info","IMAGE","SAMPLER",) 
    FUNCTION = "node"

    CATEGORY = chaosaiart_higher.name("animation")

    @staticmethod
    def vae_encode_crop_pixels(pixels):
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
        return pixels
    
    def node(self,model,Img2img_input_Size, positive, negative, denoise, seed_start, vae, Image_Mode, activ_frame,
             Image_Size, steps,seed_mode, cfg, sampler_name, scheduler, start_Image = None, zoom_frame = None):
            
        seedMode = {
            "fixed - 0.50 - SD1.5":0.5, 
            "fixed - 0.45":0.45, 
            "fixed - 0.48":0.48,  
            "fixed - 0.52":0.52,  
            "fixed - 0.55":0.55
        }
        sizeMode = { "360p":"360p", "480p":"480p", "HD":"HD", "Full HD":"Full_HD" }
        screenMode = { "Widescreen / 16:9":"widht", "Portrait (Smartphone) / 9:16":"height", "Widht = Height":"widht=height" } 
        infoSize, batch_height, batch_width, height, width = chaosaiart_higher.emptyVideoSize(screenMode[Image_Mode],sizeMode[Image_Size])   

        info = ""    

        if (self.last_activ_frame >= activ_frame):
            self.started    = False
            self.cache_seed = -1
            self.cache_seed_2 = -1
            self.img2video_mode = True
        
        latent = None
        newLatent = False
        if self.started and self.cache_latent is not None:
            info += "Use Cache Image\n"
            latent = self.cache_latent
        else:    
            if start_Image is not None:    
                
                if Img2img_input_Size == "resize":
                    start_Image = chaosaiart_higher.resize_image_pil(start_Image,batch_width,batch_height)
                    
                if Img2img_input_Size == "crop": 
                    start_Image = chaosaiart_higher.resize_image_crop_pil(start_Image,batch_width,batch_height)

                pixels = self.vae_encode_crop_pixels(start_Image)
                latent = vae.encode(pixels[:,:,:,:3])
                info += "It's frame-after-frame animation.\nPress 'Queue Prompt' for each new frame or use 'Batch count' in 'Extras options'.\n"
                  
        if latent is None: 
            newLatent = True
            denoise, batch_size = 1, 1 
            latent = torch.zeros([batch_size, 4, batch_height // 8, batch_width // 8], device=self.device)
            info += "It's frame-after-frame animation.\nPress 'Queue Prompt' for each new frame or use 'Batch count' in 'Extras options'.\n"
            self.img2video_mode = False

        latent_image = {"samples":latent} 
        
        disable_noise = False
        force_full_denoise = True
        
        seed = seed_start
        seed_2 = -1
        start_at_step = 0
        end_at_step = steps
        
        seed, self.cache_seed  = chaosaiart_higher.txt2video_SEED_cachSEED(seed, self.cache_seed)

        if seed_mode == "increment":
            
            #Increment Seed
            seed = seed + activ_frame - 1 
            samples =  chaosaiart_higher.ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)
        
            info += f"Frame: {activ_frame}\n" +f"activ_seed: {seed}\nStart_Seed: {self.cache_seed}"
        else:
  
            if newLatent:  
                seed_2, self.cache_seed_2 = chaosaiart_higher.txt2video_SEED_cachSEED(seed_2, self.cache_seed_2) 
                samples =  chaosaiart_higher.ksampler(model, seed_2, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)
                latent_image = samples[0] 
            
            #Fixed Seed
            splitt_factor = seedMode[seed_mode]
            splittStep = int(chaosaiart_higher.round(steps * splitt_factor)) 

            start_at_step, end_at_step = 0, splittStep
            force_full_denoise = False 
            samples =  chaosaiart_higher.ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)
            
            latent_image = samples[0]
            start_at_step, end_at_step = splittStep, steps
            denoise, disable_noise  = 1, True 
            samples =  chaosaiart_higher.ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)
            
            info += f"Frame: {activ_frame}\n" +f"activ_seed: {seed}\n"
            info += "" if self.img2video_mode else f"txt2video mode actived - first img seed: {self.cache_seed_2}\n(For First IMG seed controll use Ksampler txt2video)\n" 
             
            
        image = vae.decode(samples[0]["samples"]) 
        self.cache_latent = samples[0]["samples"]
        self.last_activ_frame = activ_frame
        self.started = True
        
        return (info, image,samples[0]) 
    
        
class chaosaiart_KSampler_a2: #txt2video
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()
        self.cache_latent = None
        self.last_activ_frame = 0
        self.started = False
        self.cache_seed = -1
        self.cache_seed_txt2video = -1
        #self.img2img_Size, self.Image_Size, self.Image_Mode =  None, None, None


    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    { 
                        "activ_frame":("ACTIV_FRAME",),
                        "model": ("MODEL",),     
                        "Image_Mode":(["Widht = Height","Widescreen / 16:9","Portrait (Smartphone) / 9:16"],),
                        "Image_Size":(["360p","480p","HD","Full HD",],), 
                        "seed_main":("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff, "step": 1}),
                        "seed_mode":([
                                    "fixed: denoise=denise || denoise=1",
                                    "increment", 
                                ],), 
                        "fixedMode_Splitte_by":("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step":0.01}),
                        "fixedMode_seed_first_Img":("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff, "step": 1}),
                        "steps": ("INT", {"default": 25, "min": 0, "max": 10000, "step": 1}),
                        "denoise":("FLOAT", {"default": 0.7, "min": 0, "max": 1, "step":0.01}),
                        "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                        "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                        "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                        "positive": ("CONDITIONING", ),
                        "negative": ("CONDITIONING", ),  
                        "vae":("VAE",),
                    },
                     "optional":{   
                        "zoom_frame":("ZOOM_FRAME",),  
                    }
                }

    RETURN_TYPES = ("STRING","IMAGE","LATENT",) 
    RETURN_NAMES = ("Info","IMAGE","SAMPLER",) 
    FUNCTION = "node"

    CATEGORY = chaosaiart_higher.name("animation")

    @staticmethod
    def vae_encode_crop_pixels(pixels):
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
        return pixels
    
    def node(self,model,vae,seed_main,fixedMode_seed_first_Img, positive, negative, denoise,
            Image_Mode, activ_frame,fixedMode_Splitte_by,
            Image_Size, steps,seed_mode, cfg, sampler_name, scheduler, zoom_frame = None):
             
        sizeMode = { "360p":"360p", "480p":"480p", "HD":"HD", "Full HD":"Full_HD" }
        screenMode = { "Widescreen / 16:9":"widht", "Portrait (Smartphone) / 9:16":"height", "Widht = Height":"widht=height" } 
        infoSize, batch_height, batch_width, height, width = chaosaiart_higher.emptyVideoSize(screenMode[Image_Mode],sizeMode[Image_Size])   

        info = ""   

        if (self.last_activ_frame >= activ_frame):
            self.started = False
            self.cache_seed = -1
            self.cache_seed_txt2video = -1
        
        latent = None
        newLatent = False
        if self.started and self.cache_latent is not None:
            info += "Use Cache Image\n"
            latent = self.cache_latent 
            
        if latent is None: 
            newLatent = True
            denoise, batch_size = 1, 1 
            latent = torch.zeros([batch_size, 4, batch_height // 8, batch_width // 8], device=self.device)
            info += "It's frame-after-frame animation.\nPress 'Queue Prompt' for each new frame or use 'Batch count' in 'Extras options'.\n"
                

        latent_image = {"samples":latent} 
        
        disable_noise = False
        force_full_denoise = True
        
        seed = seed_main
        seed_txt2video = fixedMode_seed_first_Img
        start_at_step = 0
        end_at_step = steps
         
        seed, self.cache_seed  = chaosaiart_higher.txt2video_SEED_cachSEED(seed, self.cache_seed)
      
        if seed_mode == "increment":
            #Increment Seed
            seed = seed + activ_frame - 1 
            samples =  chaosaiart_higher.ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)
        
            info += f"Frame: {activ_frame}\n" +f"activ_seed: {seed}\nStart_Seed: {self.cache_seed}"
        else: 
            if newLatent:  
                seed_txt2video, self.cache_seed_txt2video = chaosaiart_higher.txt2video_SEED_cachSEED(seed_txt2video, self.cache_seed_txt2video) 
                samples =  chaosaiart_higher.ksampler(model, seed_txt2video, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)
                latent_image = samples[0] 
 
            #Fixed Seed
            splitt_factor = fixedMode_Splitte_by
            splittStep = int(chaosaiart_higher.round(steps * splitt_factor)) 

            start_at_step, end_at_step = 0, splittStep
            force_full_denoise = False 
            samples =  chaosaiart_higher.ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)
            
            latent_image = samples[0]
            start_at_step, end_at_step = splittStep, steps
            denoise, disable_noise  = 1, True 
            samples =  chaosaiart_higher.ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)
            
            info += f"Frame: {activ_frame}\n" +f"activ_seed: {seed}\nFirst Image SEED: {self.cache_seed_txt2video}"
           
        image = vae.decode(samples[0]["samples"]) 
        self.cache_latent = samples[0]["samples"]
        self.last_activ_frame = activ_frame
        self.started = True
        
        return (info, image,samples[0]) 
    
class chaosaiart_KSampler_a1a: #txt2video & img2video
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()
        self.cache_latent = None
        self.last_activ_frame = 0
        self.started = False
        self.cache_seed = -1
        self.cache_seed_2 = -1
        self.cache_seed_txt2video = -1
        self.img2video_mode = True
        #self.img2img_Size, self.Image_Size, self.Image_Mode =  None, None, None
    

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    { 
                        "activ_frame":("ACTIV_FRAME",),
                        "model": ("MODEL",),     
                        "Image_Mode":(["Widht = Height","Widescreen / 16:9","Portrait (Smartphone) / 9:16"],),
                        "Image_Size":(["360p","480p","HD","Full HD",],),
                        "Img2img_input_Size":(["resize","crop","override"],),
                        "txt2video_seed_First_IMG":("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff, "step": 1}),
                        "seed_activ":("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff, "step": 1}),
                        "seed_mode":(["fixed || fixed","increment || increment","fixed || increment","increment || fixed"],),
                        "splitt_by_steps":("INT", {"default": 12, "min": 1, "max": 9999, "step":1}),
                        "denoise_part_1":("FLOAT", {"default": 0.6, "min": 0, "max": 1, "step":0.01}),
                        "denoise_part_2":("FLOAT", {"default": 1, "min": 0, "max": 1, "step":0.01}),
                        "steps": ("INT", {"default": 25, "min": 0, "max": 10000, "step": 1}),
                        "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                        "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                        "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                        "positive": ("CONDITIONING", ),
                        "negative": ("CONDITIONING", ), 
                        "vae": ("VAE",),
                    },
                     "optional":{  
                        "start_Image":("IMAGE",),  
                        "zoom_frame":("ZOOM_FRAME",),  
                    }
                }

    RETURN_TYPES = ("STRING","IMAGE","LATENT",) 
    RETURN_NAMES = ("Info","IMAGE","SAMPLER",) 
    FUNCTION = "node"

    CATEGORY = chaosaiart_higher.name("animation")

    @staticmethod
    def vae_encode_crop_pixels(pixels):
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
        return pixels
    
    def node(self,model, positive, negative, activ_frame,vae, 
            Image_Mode, Image_Size,Img2img_input_Size, 
            txt2video_seed_First_IMG,seed_activ,splitt_by_steps,seed_mode,
            denoise_part_1, denoise_part_2,   
            steps,cfg, sampler_name, scheduler, start_Image = None, zoom_frame = None):
            
        sizeMode = { "360p":"360p", "480p":"480p", "HD":"HD", "Full HD":"Full_HD" }
        screenMode = { "Widescreen / 16:9":"widht", "Portrait (Smartphone) / 9:16":"height", "Widht = Height":"widht=height" } 

        if (self.last_activ_frame >= activ_frame):
            self.started = False
            self.cache_seed = -1
            self.cache_seed_2 = -1
            self.cache_seed_txt2video = -1 
            self.img2video_mode = True
        
        infoSize, batch_height, batch_width, height, width = chaosaiart_higher.emptyVideoSize(screenMode[Image_Mode],sizeMode[Image_Size])   
 
        info = ""  
        
        latent = None
        newLatent = False
        if self.started and self.cache_latent is not None:
            info += "Use Cache Image\n"
            latent = self.cache_latent
        else:    
            if start_Image is not None:    
                
                if Img2img_input_Size == "resize":
                    start_Image = chaosaiart_higher.resize_image_pil(start_Image,batch_width,batch_height)
                    
                if Img2img_input_Size == "crop": 
                    start_Image = chaosaiart_higher.resize_image_crop_pil(start_Image,batch_width,batch_height)
                    
                pixels = self.vae_encode_crop_pixels(start_Image)
                latent = vae.encode(pixels[:,:,:,:3])
                info += "It's frame-after-frame animation.\nPress 'Queue Prompt' for each new frame or use 'Batch count' in 'Extras options'.\n"
         
        if latent is None: 
            newLatent = True
            denoise, batch_size = 1, 1 
            latent = torch.zeros([batch_size, 4, batch_height // 8, batch_width // 8], device=self.device)
            info += "It's frame-after-frame animation.\nPress 'Queue Prompt' for each new frame or use 'Batch count' in 'Extras options'.\n"
            self.img2video_mode = False
            
        latent_image = {"samples":latent} 
        
        start_at_step = 0
        end_at_step = steps
        disable_noise = False
        force_full_denoise = True
        
        seed = seed_activ
        seed_2 = seed_activ
        seed_txt2video = txt2video_seed_First_IMG 
        seed, self.cache_seed  = chaosaiart_higher.txt2video_SEED_cachSEED(seed, self.cache_seed)
        seed_2, self.cache_seed_2 =seed ,seed
      
        parts = seed_mode.split(" || ")
        if "increment" in parts[0]:
            seed = seed + activ_frame - 1 
        if "increment" in parts[1]:
            seed_2 = seed_2 + activ_frame - 1
            
        if newLatent: 
            seed_txt2video, self.cache_seed_txt2video  = chaosaiart_higher.txt2video_SEED_cachSEED(seed_txt2video, self.cache_seed_txt2video)
            info += f"txt2video Mode - Fixed Seed :\n First Frame: new Image Seed Generated \n"
            samples =  chaosaiart_higher.ksampler(model, seed_txt2video, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)
            latent_image = samples[0] 

        #splitt_factor = splitt_by
        splittStep = splitt_by_steps

        start_at_step, end_at_step = 0, splittStep
        denoise, force_full_denoise = denoise_part_1, False 
        samples =  chaosaiart_higher.ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)
        
        if steps > splittStep:
            latent_image = samples[0]
            start_at_step, end_at_step = splittStep, steps
            denoise, disable_noise  = denoise_part_2, True 
            samples =  chaosaiart_higher.ksampler(model, seed_2, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)
        
        
        info += f"Frame: {activ_frame}\n\n" 
        info += f"Start_seed Part_1: {self.cache_seed},\nActiv_seed Part_1: {seed},\n\n"
        info += f"Start_seed Part_2: {self.cache_seed},\nActiv_seed Part_2: {seed_2}\n\n"
        info += "" if self.img2video_mode else f"txt2video mode actived - first img seed: {self.cache_seed_txt2video}\n" 
       
        image = vae.decode(samples[0]["samples"]) 
        self.cache_latent = samples[0]["samples"]
        self.last_activ_frame = activ_frame
        self.started = True
        
        return (info, image,samples[0]) 
    


class chaosaiart_Ksampler_attribut:
    def __init__(self): 
        self.started = False

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {      
                        "steps": ("INT", {"default": 25, "min": 0, "max": 10000}),
                        "Synchro_denoise":(["No","All = First 🔶KSampler Splitted"],),
                        "Synchro_seed":(["No","All = First 🔶KSampler Splitted"],),
                        "Synchro_cfg":(["No","All = First 🔶KSampler Splitted"],),
                        "Synchro_sampler":(["No","All = First 🔶KSampler Splitted"],),
                        "Synchro_scheduler":(["No","All = First 🔶KSampler Splitted"],),   
                        "latent":("LATENT",),  
                    }, 
                }

    RETURN_TYPES = ("K_ATTRIBUT", "LATENT",) 
    RETURN_NAMES = ("MAIN_K_ATTRIBUT", "LATENT",) 
    FUNCTION = "node"

    CATEGORY = chaosaiart_higher.name("animation")
  
    @staticmethod
    def vae_encode_crop_pixels(pixels):
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
        return pixels
    
    def node(self, 
             Synchro_denoise,Synchro_seed,Synchro_cfg,Synchro_sampler,Synchro_scheduler,
             latent, steps, #Image_Mode, Image_Size, Img2img_input_Size, vae_by_imageInput = None, image = None
             ):
        
        Synchro_denoise = -1 if Synchro_denoise == "No" else 0
        Synchro_seed = -1 if Synchro_seed == "No" else 0
        Synchro_cfg = -1 if Synchro_cfg == "No" else 0
        Synchro_sampler = -1 if Synchro_sampler == "No" else 0
        Synchro_scheduler = -1 if Synchro_scheduler == "No" else 0
    
        out = {   
            "last_end_step":0, 
            "steps":steps,
            "denoise": Synchro_denoise,
            "seed": Synchro_seed,
            "cfg": Synchro_cfg,
            "sampler_name": Synchro_sampler,
            "scheduler": Synchro_scheduler
        }
        
        return out, latent, 


class chaosaiart_KSampler_expert_1:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":
            { 
                "model": ("MODEL",),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "k_attribut": ("K_ATTRIBUT",),
                "latent":("LATENT",), 
                "seed":("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff, "step": 1}), 
                "end_at_step": ("INT", {"default": 25, "min": 0, "max": 10000}),
                "denoise":("FLOAT", {"default": 1, "min": 0, "max": 1}), 
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
            }, 
        }

 
    RETURN_TYPES = ("K_ATTRIBUT","LATENT","STRING",) 
    RETURN_NAMES = ("SUB_K_ATTRIBUT","SAMPLER","Info") 
    FUNCTION = "node"

    CATEGORY = chaosaiart_higher.name("animation")
    
    def node( self, model, positive, negative, k_attribut, 
             latent,seed, end_at_step, denoise, cfg,sampler_name,scheduler ):
        info = ""
        
        start_at_step = k_attribut["last_end_step"] 
        steps = k_attribut["steps"] 
        
        
        if k_attribut["denoise"] > 0:
            denoise = k_attribut["denoise"]    
        else: 
            if k_attribut["denoise"] == 0:
                k_attribut["denoise"] = denoise 

        if k_attribut["seed"] > 0:
            seed = k_attribut["seed"]
        else:     
            if k_attribut["seed"] == 0:
                k_attribut["seed"] = seed

        if k_attribut["cfg"] > 0:
            cfg = k_attribut["cfg"]
        else:    
            if k_attribut["cfg"] == 0:
                k_attribut["cfg"] = cfg

        if k_attribut["sampler_name"] == -1 or k_attribut["sampler_name"] == 0:
            if k_attribut["sampler_name"] == 0:
                k_attribut["sampler_name"] = sampler_name
        else:        
            sampler_name = k_attribut["sampler_name"]

        if k_attribut["scheduler"] == -1 or k_attribut["scheduler"] == 0: 
            if k_attribut["scheduler"] == 0:
                k_attribut["scheduler"] = scheduler 
        else:
            scheduler = k_attribut["scheduler"]
        
 
        latent_image = latent

        disable_noise = True 
        force_full_denoise = False
        if start_at_step == 0:             
            disable_noise = False
            force_full_denoise = False
        #last -> force_full_denoise = True ?
     
        samples =  chaosaiart_higher.ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)
        
        out_k_attribut = {
            "last_end_step": end_at_step,  # +1 ?
            "steps": k_attribut["steps"],
            "denoise": k_attribut["denoise"], 
            "seed": k_attribut["seed"], 
            "cfg": k_attribut["cfg"], 
            "sampler_name": k_attribut["sampler_name"], 
            "scheduler": k_attribut["scheduler"] 
        }  
 
        info += f"Start_at_step: {start_at_step}\nend_at_step: {end_at_step}\nsteps: {steps}\ndenoise: {denoise}\n"  
        info += f"seed: {seed}\ncfg: {cfg}\nsampler_name: {sampler_name}\nscheduler: {scheduler}"

        return out_k_attribut, samples[0], info,
        
     
            
 


class chaosaiart_forPreview:
    
    def __init__(self):
        self.image_batch = None
        self.count = 1

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {
                    "restart":("RESTART",),
                    "image":("IMAGE",),
                    "Preview_Max": ("INT", {"default": 50, "min": 0, "max": 10000}), 
                }
            }

    RETURN_TYPES = ("IMAGE",) 
    RETURN_NAMES = ("IMAGE",) 
    FUNCTION = "node"

    CATEGORY = chaosaiart_higher.name("animation")

    def node(self, image, restart, Preview_Max):

        if self.image_batch == None or restart >= 1 or self.count > Preview_Max:
            self.image_batch = image
            self.count = 2
            return image,  

        self.count += 1
        imageBatch = torch.cat([self.image_batch,image], dim=0)
        self.image_batch = imageBatch 
        return imageBatch,
    
     
class chaosaiart_KSampler5:

    def __init__(self):
        self.device  = comfy.model_management.intermediate_device()
        self.counter = 1
        self.reloader_Num = 0 

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                        "restart": ("RESTART",),
                        "model": ("MODEL",),  
                        "Image_Mode":(["Widht = Height","Widescreen / 16:9","Portrait (Smartphone) / 9:16"],),
                        "Image_Size":(["360p","480p","HD","Full HD",],),
                        "Img2img_input_Size":(["resize","crop","override"],),"seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                        "steps": ("INT", {"default": 25, "min": 1, "max": 10000}),
                        "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                        "end_at_step": ("INT", {"default": 25, "min": 0, "max": 10000}),
                        "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                        "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                        "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                        "positive": ("CONDITIONING", ),
                        "negative": ("CONDITIONING", ), 
                        "vae": ("VAE", ),
                     },
                     "optional":{
                        "Start_Image_Override": ("IMAGE", ), 
                    }
                }

    RETURN_TYPES = ("STRING","IMAGE","LATENT",) 
    RETURN_NAMES = ("Info","IMAGE","SAMPLES",) 
    FUNCTION = "node"

    CATEGORY = chaosaiart_higher.name("ksampler")

    @staticmethod
    def vae_encode_crop_pixels(pixels):
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
        return pixels
    
    def node(self, 
             model, seed, steps, cfg, sampler_name, scheduler, positive, negative, start_at_step, end_at_step, 
             vae,Image_Mode,Image_Size,Img2img_input_Size,
             restart=0 , Start_Image_Override = None,
             denoise=1.0):
        
        start_at_stepNum = start_at_step 
        return_with_leftover_noise = "disable"
        empty_Img_batch_size = 1
        add_noise = "enable"
        
        info = ""
        info += "It's frame-after-frame animation.\nPress 'Queue Prompt' for each new frame or use 'Batch count' in 'Extras options'.\n"
        
        sizeMode = { "360p":"360p", "480p":"480p", "HD":"HD", "Full HD":"Full_HD" }
        screenMode = { "Widescreen / 16:9":"widht", "Portrait (Smartphone) / 9:16":"height", "Widht = Height":"widht=height" } 
        infoSize, batch_height, batch_width, height, width = chaosaiart_higher.emptyVideoSize(screenMode[Image_Mode],sizeMode[Image_Size])   
             
        if self.counter == 1 or restart >= 1:
            self.counter = 1
            if Start_Image_Override is not None:
                
                if Img2img_input_Size == "resize":
                    Start_Image_Override = chaosaiart_higher.resize_image_pil(Start_Image_Override,batch_width,batch_height)
                    
                if Img2img_input_Size == "crop": 
                    Start_Image_Override = chaosaiart_higher.resize_image_crop_pil(Start_Image_Override,batch_width,batch_height)
                
                #img -> Vae -> Latent  
                pixels = Start_Image_Override
                pixels = self.vae_encode_crop_pixels(pixels)
                t = vae.encode(pixels[:,:,:,:3])
                latent_image = {"samples":t} 
                info += "- Restarted: with Start_Image -\n" 
            else: 
                start_at_stepNum = 0
                latent = torch.zeros([empty_Img_batch_size, 4, batch_height // 8, batch_width // 8], device=self.device)
                latent_image = {"samples":latent} 
                info += "- Restarted: with a Empty Image -\n" 
        else:  
            saved_latent = chaosaiart_higher.reloader_x("ksampler",self.reloader_Num,False,None)
            latent_image = {"samples":saved_latent}  
            info += "- continued -\n" 

        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True 

        samples =  chaosaiart_higher.ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise, disable_noise=disable_noise, start_step=start_at_stepNum, last_step=end_at_step, force_full_denoise=force_full_denoise)
         
        info += "Created Frame = self.counter" 
        self.counter += 1 
        info += "if you want more Controll use gets the Advenden Worfklow: ... "
        
        chaosaiart_higher.reloader_x("ksampler",self.reloader_Num,True,samples[0]["samples"])   
        image = vae.decode(samples[0]["samples"])
        return (info,image,samples[0],)
    
 
class chaosaiart_SaveImage:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4
        self.num = 0
        self.started = False 

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ), 
                "filename_prefix": ("STRING", {"default": "chaosaiart"})
                },
            "optional": {
                "restart": ("RESTART",),
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "node"

    OUTPUT_NODE = True

    CATEGORY = chaosaiart_higher.name("main")

    def node(self, images,restart=0, filename_prefix="chaosaiart", prompt=None, extra_pnginfo=None):
        if restart >= 1 or self.started == False:
            self.started = True 
            print(self.output_dir)
            print(f"{filename_prefix}/v_{self.num:04d}")
            while os.path.exists(os.path.join(self.output_dir, filename_prefix, f"v_{self.num:04d}")):
                self.num += 1
            #while os.path.exists(f"{filename_prefix}/v_{self.num:04d}"):
                #self.num += 1

        #filename_prefix += "/v_"+"{:04d}".format(self.num)+"/i_"
        filename_prefix += f"/v_{self.num:04d}/img_"  
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            file = f"{filename}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return { "ui": { "images": results } }
 
#FIXME: TODO: NOTIZ:
#FIXME: TODO: NOTIZ:
#FIXME: TODO: NOTIZ: Error sometimes, load to early , i thing its come with when is the Node Created.
#First Time see this bug, by Adding: OUTPUT_NODE = True, maybe its a part of it.
class chaosaiart_reloadIMG_Save:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "reloader": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            },
        }
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
    
    
    OUTPUT_NODE = True

    RETURN_TYPES = ("IMAGE","STRING",)
    RETURN_NAMES = ("IMAGE","Info",)
    FUNCTION = "node"
    CATEGORY = chaosaiart_higher.name("cache")
 
    def node(self, image, reloader):  
        chaosaiart_higher.reloader_x("img",reloader,True,image) 
        info = "this Node only save for Chaosaiart -> Reload function"
        return(image,str(info),)
    

class chaosaiart_reloadIMG_Load: 
    def __init__(self): 
        self.is_Started = False
        self.device = comfy.model_management.intermediate_device()
         
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "restart":("RESTART",),
                "reloader": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}), 
                "preSave_Image_Mode":(["Widescreen / 16:9","Portrait (Smartphone) / 9:16","Widht = Height"],),
                "preSave_Image_Size":(["360p","480p","HD","Full HD",],),
            },
            "optional":{ 
                "preSave_image_override":("IMAGE",),   
                "vae": ("VAE", ),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "node"
    CATEGORY = chaosaiart_higher.name("cache")
 
    def node(self,reloader,preSave_Image_Size,preSave_Image_Mode,preSave_image_override=None,restart=0,vae=None):  
 
        if restart == 0 and self.is_Started: 
            return chaosaiart_higher.reloader_x("img",reloader,False,None),
            
        self.is_Started = True 
        if preSave_image_override is not None:
            return preSave_image_override,
            
        if vae is None: 
            raise ValueError(chaosaiart_higher.name("main")+" - Cache Reloader IMG-> LOAD need VAE without preSave_image_override")
            
        sizeMode = {
            "360p":"360p",
            "480p":"480p",#4px heigher
            "HD":"HD",
            "Full HD":"Full_HD", 
        } 
        screenMode = {
            "Widescreen / 16:9":"widht",
            "Portrait (Smartphone) / 9:16":"height",
            "Widht = Height":"widht=height" 
        }  
        info, batch_height, batch_width, height, width = chaosaiart_higher.emptyVideoSize(screenMode[preSave_Image_Mode],sizeMode[preSave_Image_Size])
        batch_size = 1   
        latent = torch.zeros([batch_size, 4, batch_height // 8, batch_width // 8], device=self.device)

        return vae.decode(latent),

class chaosaiart_reloadLatent_Save:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "reloader": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            },
        }
    
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
    
    
    OUTPUT_NODE = True

    RETURN_NAMES = ("LATENT","Info",)
    RETURN_TYPES = ("LATENT","STRING",)
    FUNCTION = "node"
    CATEGORY = chaosaiart_higher.name("cache")
 
    def node(self, latent, reloader): 
        chaosaiart_higher.reloader_x("latent",reloader,True,latent) 
        info = "this Node only save for Chaosaiart -> Reload function"
        return(latent,str(info),)
    
    
    
class chaosaiart_reloadLatent_Load: 
    def __init__(self): 
        self.is_Started = False 
        self.device = comfy.model_management.intermediate_device()
         
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "restart":("RESTART",), 
                "reloader": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "preSave_Image_Mode":(["Widescreen / 16:9","Portrait (Smartphone) / 9:16","Widht = Height"],),
                "preSave_Image_Size":(["360p","480p","HD","Full HD",],)
            },
            "optional":{ 
                "preSave_Latent_override": ("LATENT",),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("LATENT",)
    FUNCTION = "node"
    CATEGORY = chaosaiart_higher.name("cache")
 
    def node(self,preSave_Image_Mode,preSave_Image_Size, reloader,restart=0, preSave_Latent_override=None): 
 
        if restart == 0 and self.is_Started: 
            return chaosaiart_higher.reloader_x("latent",reloader,False,None),
            
        self.is_Started = True 
        if preSave_Latent_override is not None:
            return preSave_Latent_override,
            
        sizeMode = {
            "360p":"360p",
            "480p":"480p",#4px heigher
            "HD":"HD",
            "Full HD":"Full_HD", 
        }
        screenMode = {
            "Widescreen / 16:9":"widht",
            "Portrait (Smartphone) / 9:16":"height",
            "Widht = Height":"widht=height" 
        } 
        info, batch_height, batch_width, height, width = chaosaiart_higher.emptyVideoSize(screenMode[preSave_Image_Mode],sizeMode[preSave_Image_Size])
        batch_size = 1   
        latent = torch.zeros([batch_size, 4, batch_height // 8, batch_width // 8], device=self.device)

        return {"samples":latent},
    
class chaosaiart_reloadAny_Save:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Any": (anyType,),
                "reloader": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            },
        }
    
    
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
    
    RETURN_NAMES = ("ANY","Info",)
    RETURN_TYPES = (anyType,"STRING",)
    FUNCTION = "node"
    CATEGORY = chaosaiart_higher.name("cache")
 
    def node(self, Any, reloader): 
        chaosaiart_higher.reloader_x("any",reloader,True,Any) 
        info = "this Node only save for Chaosaiart -> Reload function"
        return(Any,str(info),)
    
class chaosaiart_reloadAny_Load: 
    
    def __init__(self): 
        self.is_Started = False
         
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "restart":("RESTART",),
                "any_pre_Save_Out":(anyType,),
                "reloader": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
    
    RETURN_TYPES = (anyType,)
    RETURN_NAMES = ("ANY",)
    FUNCTION = "node"
    CATEGORY = chaosaiart_higher.name("cache")
 
    def node(self,reloader,any_pre_Save_Out,restart=0): 

        if restart == 1 or self.is_Started == False:
            self.is_Started = True
            out = any_pre_Save_Out
        else:
            out = chaosaiart_higher.reloader_x("any",reloader,False,None)
            if out is None:
                print("Chaosaiart - reloader: Load Failed")
                out = any_pre_Save_Out
        return ( out, ) 



class chaosaiart_TextCLIPEncode_simple:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "forceInput": True}), 
                "clip": ("CLIP", )
                }
            }
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "node"

    CATEGORY = chaosaiart_higher.name("prompt")
    def node(self, clip, text):
        return (chaosaiart_higher.textClipEncode(clip,text), )



class chaosaiart_TextCLIPEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positiv_txt": ("STRING", {"multiline": True, "forceInput": True}),
                "negativ_txt": ("STRING", {"multiline": True, "forceInput": True}),
                "clip": ("CLIP", )
            }}
    RETURN_TYPES = ("CONDITIONING","CONDITIONING",)
    RETURN_NAMES = ("POSITIV","NEGATIV",)
    FUNCTION = "node"

    CATEGORY = chaosaiart_higher.name("prompt")
    def node(self, clip, positiv_txt,negativ_txt):
        return (chaosaiart_higher.textClipEncode(clip,positiv_txt),chaosaiart_higher.textClipEncode(clip,negativ_txt), )


class chaosaiart_TextCLIPEncode_lora:
    def __init__(self):  
        self.lora_cache = []
        self.cache_lora_feed = []

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model":("MODEL",),
                "positiv_txt": ("STRING", {"multiline": True, "forceInput": True}),
                "negativ_txt": ("STRING", {"multiline": True, "forceInput": True}),
                "clip": ("CLIP",),
                "lora":("LORA",),
            }}
    RETURN_TYPES = ("MODEL","CONDITIONING","CONDITIONING",)
    RETURN_NAMES = ("MODEL","POSITIV","NEGATIV",)
    FUNCTION = "node"

    CATEGORY = chaosaiart_higher.name("prompt")
    def node(self, clip, positiv_txt,negativ_txt,model,lora): 
   
        loraArray = lora
        out_Model, positiv_clip, negativ_clip, self.lora_cache, lora_Info  = chaosaiart_higher.load_lora_by_Array(loraArray,model,clip,self.lora_cache)
        
        out_Positiv = chaosaiart_higher.textClipEncode(positiv_clip,positiv_txt)
        out_Negaitv = chaosaiart_higher.textClipEncode(negativ_clip,negativ_txt)
 
        return (out_Model,out_Positiv,out_Negaitv,)


class chaosaiart_MainPromptCLIPEncode:
    def __init__(self):  
        self.lora_cache = []

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model":("MODEL",), 
                "clip": ("CLIP",), 
                "main_prompt": ("MAIN_PROMPT",)
            }}
    RETURN_TYPES = ("MODEL","CONDITIONING","CONDITIONING",)
    RETURN_NAMES = ("MODEL","POSITIV","NEGATIV",)
    FUNCTION = "node"

    CATEGORY = chaosaiart_higher.name("prompt")
    def node(self,model, clip, main_prompt): 

        positiv_txt = main_prompt[0]
        negativ_txt = main_prompt[1]
        lora        = main_prompt[2]

        loraArray = lora
        out_Model, positiv_clip, negativ_clip, self.lora_cache, lora_Info  = chaosaiart_higher.load_lora_by_Array(loraArray,model,clip,self.lora_cache)
        
        out_Positiv = chaosaiart_higher.textClipEncode(positiv_clip,positiv_txt)
        out_Negaitv = chaosaiart_higher.textClipEncode(negativ_clip,negativ_txt)
         
        return (out_Model,out_Positiv,out_Negaitv,)
    
class chaosaiart_FramePromptCLIPEncode:
    def __init__(self):  
        self.lora_cache = []

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model":("MODEL",), 
                "clip": ("CLIP",), 
                "frame_prompt": ("FRAME_PROMPT",)
            }}
    RETURN_TYPES = ("MODEL","CONDITIONING","CONDITIONING",)
    RETURN_NAMES = ("MODEL","POSITIV","NEGATIV",)
    FUNCTION = "node"

    CATEGORY = chaosaiart_higher.name("prompt")
    def node(self,model, clip, frame_prompt): 

        positiv_txt = frame_prompt[1][0]
        negativ_txt = frame_prompt[1][1]
        lora        = frame_prompt[1][2]

        loraArray = lora

        out_Model, positiv_clip, negativ_clip, self.lora_cache, lora_Info  = chaosaiart_higher.load_lora_by_Array(loraArray,model,clip,self.lora_cache)
        
        out_Positiv = chaosaiart_higher.textClipEncode(positiv_clip,positiv_txt)
        out_Negaitv = chaosaiart_higher.textClipEncode(negativ_clip,negativ_txt)
         
        return (out_Model,out_Positiv,out_Negaitv,)
 



#TODO: Testen
class chaosaiart_restarter_advanced:
    def __init__(self):  
        self.Version = 0
        self.counter_x = 0
        self.startNum = 0
        self.stopNum = 0 
        self.mode_1 = ""
        self.mode_2 = "" 
        self.at_end = False
        self.End_OneTime = False
        self.started = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Mode":(["Loop___Restart_at_End","Endless___1x_Restart_at_End","stop_at_End___1x_Restart_at_End","stop_at_End___no_Restart","Endless___no_Restart"],),
                "Start":("INT", {"default": 1, "min": 1, "max": 18446744073709551615, "step": 1}),
                "End": ("INT", {"default": 100, "min": 0, "max": 18446744073709551615, "step": 1}),
                "Version": ("INT", {"default": 0, "min": 0, "max": 18446744073709551615, "step": 1}),
            },
            "optional": { 
                "restart":("RESTART",),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
    
    RETURN_TYPES = ("STRING","ACTIV_FRAME", "RESTART","REPEAT",)
    RETURN_NAMES = ("Info","ACTIV_FRAME", "RESTART","REPEAT",)

    FUNCTION = "node"

    CATEGORY = chaosaiart_higher.name("restart")
  
    def node(self, Mode, Start, End, Version, restart=0):
       
        start_Num = Start
        stop_Num  = End
        step_Num  = 1 
        reset_Num = restart
        reset = False

        restartOUT = 0
        
        mode_1 = Mode
        mode_2 = 'increment' if start_Num < stop_Num else 'decrement' 
        
        if not self.Version == Version or reset_Num >= 1:
            reset = True
            print("Chaosaiart - restarter_advanced: restarted by Next Version")
        
        if not self.mode_1 == mode_1 or not self.mode_2 == mode_2 or not self.startNum == start_Num or stop_Num == start_Num:
            reset = True
            print("Chaosaiart - restarter_advanced: restarted by changing input")
            
        if self.at_end == True: 
            if mode_1 == "Loop___Restart_at_End": 
                reset = True
                print("Chaosaiart - restarter_advanced: restarted End, loop started")
            if mode_1 == "stop_at_End___1x_Restart_at_End" or mode_1 == "Endless___1x_Restart_at_End": 
                if self.End_OneTime == False: 
                    self.End_OneTime = True
                    restartOUT = 1 
                    print("Chaosaiart - restarter_advanced: restarted End")


        if reset:
            self.End_OneTime = False 
            self.started = False
            restartOUT = 1

        counter = start_Num  
        self.at_end = False 
 
        if self.started == True: 
            counter = self.counter_x + step_Num if mode_2 == 'increment' else self.counter_x - step_Num

            if counter >= stop_Num: 
                if mode_2 == 'increment': 
                    self.at_end = True 
                    if mode_1 == "stop_at_End___1x_Restart_at_End" or mode_1 == "stop_at_End___no_Restart":
                        counter = stop_Num 
            if counter <= stop_Num:  
                if not mode_2 == 'increment': 
                    self.at_end = True  
                    if mode_1 == "stop_at_End___1x_Restart_at_End" or mode_1 == "stop_at_End___no_Restart": 
                        counter = stop_Num

            counter = counter if counter >= 0 else 0
 
        self.mode_1 = mode_1
        self.mode_2 = mode_2 
        self.Version = Version
        self.started = True
        self.startNum = start_Num
        self.stopNum = stop_Num
        self.mode_1 = mode_1
        self.mode_2 = mode_2
        self.counter_x = counter

        if mode_2 == 'increment':
            countdown = stop_Num - counter
            repeat = stop_Num - start_Num +1 
        else:
            countdown = start_Num - counter
            repeat = start_Num - stop_Num +1
    
        info = f"Activ_Frame: {counter}\nCountdown: {countdown}\nRepeat: {repeat}"
        if restartOUT == 1:
            info += "\n\n--- is Restarted ---"  
            print("reset ------- reset")
            print("reset ------- reset")
            print("reset ------- reset")
        
        info += "\nIf you change the version parameter, all nodes connected via restart will be restarted (restart = 1), and the Activ_Frame will be reset to the start (Activ_Frame = 1)." 
    
        return ( info,int(counter), restartOUT, repeat,)    

 
class chaosaiart_restarter:
    def __init__(self):  
        self.Version = 0
        self.count = 0

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { 
                "Version": ("INT", {"default": 0, "min": 0, "max": 18446744073709551615, "step": 1}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
    
    RETURN_TYPES = ("ACTIV_FRAME", "RESTART","STRING",)
    RETURN_NAMES = ("ACTIV_FRAME", "RESTART","Info",)

    FUNCTION = "node"

    CATEGORY = chaosaiart_higher.name("restart")
 

    def node(self, Version):
        
        out_NumCheck = 0
        info = f"Restart = No\n"
        if not self.Version == Version:
            self.Version = Version
            out_NumCheck = 1
            self.count = 0
            info = f"Restart = Yes\n"

        
        self.count += 1
        info = f"Activ Frame {self.count}\n" + info
        info += "\nIf you change the version parameter, all nodes connected via restart will be restarted (restart = 1), and the Activ_Frame will be reset to the start (Activ_Frame = 1)." 
    
        return (self.count, out_NumCheck,info,)

 


class chaosaiart_any_array2input_all_small:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "array":("ARRAY",),
            }, 
        }
    
    RETURN_TYPES = (anyType,anyType,anyType,anyType,)
    RETURN_NAMES = ("ANY_1","ANY_2","ANY_3","ANY_4",)

    FUNCTION = "node"
    CATEGORY = chaosaiart_higher.name("switch")
 

    def node(self,array=None):  
        if array:
            return(array.saveItem)
        
        print("ChaosAiArt: array2input_ALL(small) no Input found.")
        return (None,)
         

class chaosaiart_any_array2input_all_big:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "array":("ARRAY",),
            }, 
        }
    
    RETURN_TYPES = (anyType,anyType,anyType,anyType,anyType,anyType,anyType,anyType,anyType,)
    RETURN_NAMES = ("ANY_1","ANY_2","ANY_3","ANY_4","ANY_5","ANY_6","ANY_7","ANY_8","ANY_9",)

    FUNCTION = "node"
    CATEGORY = chaosaiart_higher.name("switch")
 

    def node(self,array=None):  
        if array:
            return(array.saveItem)
        
        print("ChaosAiArt: array2input_ALL(big) no Input found.")
        return (None,) 


class chaosaiart_any_array2input_1Input:
    def __init__(self):  
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_nr": ("INT", {"default": 1, "min": 1, "max": 9, "step": 1}),
                "array":("ARRAY",),
            },
            "optional": {
                "input_nr_override": ("INT", {"forceInput": True}),
            }
        }
    
    RETURN_TYPES = (anyType,)
    RETURN_NAMES = ("ANY",)

    FUNCTION = "node"
    CATEGORY = chaosaiart_higher.name("switch")
 

    def node(self,array,input_nr,input_nr_override=None):  
        nr = input_nr_override if input_nr_override else input_nr
       
        if array:
            arrayInUse = array.saveItem
            if nr > len(arrayInUse):
                print("ChaosAiArt: array2input_1- nr:"+str(nr)+" not found.")
                return (None,)
            return(arrayInUse[nr-1],)
        
        print("ChaosAiArt: array2input_1Input no Input found.")
        return (None,)
     
    
class chaosaiart_any_input2array_small:
    def __init__(self):  
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_1": (anyType,),
            },
            "optional": {
                "input_2": (anyType,),
                "input_3": (anyType,),
                "input_4": (anyType,)
            }
        }
     
     
    RETURN_TYPES = ("ARRAY",)
    RETURN_NAMES = ("ARRAY",)

    FUNCTION = "node"
    CATEGORY = chaosaiart_higher.name("switch")
 

    def node(self,  
                input_1=None, 
                input_2=None, 
                input_3=None, 
                input_4=None
                ):

        array = [
            input_1,
            input_2,
            input_3,
            input_4
            ] 
        out = anySave(0,array)
        return (out,)

class chaosaiart_any_input2array_big:
    def __init__(self):  
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_1": (anyType,),
            },
            "optional": {
                "input_2": (anyType,),
                "input_3": (anyType,),
                "input_4": (anyType,),
                "input_5": (anyType,),
                "input_6": (anyType,),
                "input_7": (anyType,),
                "input_8": (anyType,),
                "input_9": (anyType,)
            }
        }
     
     
    RETURN_TYPES = ("ARRAY",)
    RETURN_NAMES = ("ARRAY",)

    FUNCTION = "node"
    CATEGORY = chaosaiart_higher.name("switch")
 

    def node(self, 
                input_1=None, 
                input_2=None, 
                input_3=None, 
                input_4=None, 
                input_5=None, 
                input_6=None, 
                input_7=None, 
                input_8=None, 
                input_9=None, 
                ):

        array = [
            input_1,
            input_2,
            input_3,
            input_4,
            input_5,
            input_6,
            input_7,
            input_8,
            input_9
            ] 
        
        out = anySave(0,array)
        return (out,)
    

#TODO: Rethinking
class chaosaiart_Any_Switch:
    def __init__(self):  
        self.Started = 0
        self.switch = 0
        self.mode = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["One Time: 1-2-2-2..", "Switch: 1-2-1-2.."],),
                "source_1": (anyType, {}),
                "source_2": (anyType, {}),
            },
            "optional": {
                "restart": ("RESTART",),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
     
    RETURN_TYPES = ("STRING",anyType,)
    RETURN_NAMES = ("Info","SOURCE_X",)

    FUNCTION = "node"
    CATEGORY = chaosaiart_higher.name("switch")
 

    def node(self, mode, source_1,source_2, restart=0):
        
        mode_num = 1
        #if mode == "2) nr:2 = 2 else 1":
            #mode_num = 2
        if mode == "Switch: 1-2-1-2..":
            mode_num = 3

       
        restart_numberNum = restart
        

        if self.mode != mode_num:
            self.mode = mode_num
            self.Started = 0

        useSourceNR = 1

        if mode_num == 1:
            if restart_numberNum == 1 or self.Started == 0: 
                self.Started = 1
                useSourceNR = 1 
            else: 
                useSourceNR = 2

        #elif mode_num == 2:
            #if restart_numberNum == 2:
                #useSourceNR = 2
            #else:
                #useSourceNR = 1    

        elif mode_num == 3:
            if restart_numberNum == 1 or self.Started == 0:
                self.Started = 1
                self.switch = 0

            if self.switch == 0:
                self.switch = 1
                useSourceNR = 1
            else:
                self.switch = 0
                useSourceNR = 2
            

        out = source_1
        if useSourceNR == 2:
            out = source_2

        info = f"Used NR: {useSourceNR}"
        return (info,out,)


class chaosaiart_Number_Switch:
    def __init__(self):  
        self.started = False
        self.switch = 0
        self.mode = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "First_IMG": ("FLOAT", {"default": 0, "min": -50, "max": 50, "step": 0.01}),
                "Rest_IMG": ("FLOAT", {"default": 1, "min": -50, "max": 50, "step": 0.01}),
            },
            "optional": {
                "restart": ("RESTART",),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
     
    RETURN_TYPES = ("INT","FLOAT","STRING",)
    RETURN_NAMES = ("INT","FLOAT","Info",)

    FUNCTION = "node"
    CATEGORY = chaosaiart_higher.name("logic")
 

    def node(self, First_IMG,Rest_IMG, restart=0):
         
        iNumber = Rest_IMG
        if restart >= 1 or self.started == False:
            self.started = True
            iNumber = First_IMG
 
        outINT = int(iNumber)
        info = f"Float: {iNumber}, INT: {outINT}"
        return (outINT,iNumber,info,)        
  
    
class chaosaiart_Denoising_Switch:
    def __init__(self):  
        self.started = False
        self.switch = 0
        self.mode = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "First_IMG": ("FLOAT", {"default": 1, "min": 0.01, "max": 1, "step": 0.01}),
                "Rest_IMG": ("FLOAT", {"default": 0.6, "min": 0.01, "max": 1, "step": 0.01}),
            },
            "optional": {
                "restart": ("RESTART",),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
     
    RETURN_TYPES = ("FLOAT","STRING",)
    RETURN_NAMES = ("DENOISE","Info",)

    FUNCTION = "node"
    CATEGORY = chaosaiart_higher.name("ksampler")
 

    def node(self, First_IMG,Rest_IMG, restart=0):
         
        iNumber = Rest_IMG
        if restart >= 1 or self.started == False:
            self.started = True
            iNumber = First_IMG
 
        info = f"Denoise: {iNumber}"
        return (iNumber,info,)

class chaosaiart_Any_Switch_small:
    def __init__(self):  
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {  
                "nr": ("INT", {"default": 0, "min": 0, "max": 1, "step": 1}),
                "source_0": (anyType, {}),
                "source_1": (anyType, {}),
            },
            "optional": { 
                "nr_override": ("INT",{"forceInput": True}), 
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
     
    RETURN_TYPES = (anyType,)
    RETURN_NAMES = ("SOURCE_X",)

    FUNCTION = "node"
    CATEGORY = chaosaiart_higher.name("switch")

    def node(self, nr, source_0, source_1, nr_override=None):
        aSource = [source_0, source_1]
        out_NR = nr_override if nr_override is not None else nr
        out_NR = 1 if out_NR >= 1 else 0 
        return aSource[out_NR],
                   
class chaosaiart_Any_Switch_Big_Number:
    def __init__(self):  
        self.Started = 0
        self.switch = 0
        self.mode = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { 
                "mode": (["nr = Source","nr = Revers", "Random"],), 
                "nr": ("INT", {"default": 0, "min": 0, "max": 9, "step": 1}),
                "source_0": (anyType, {}),
            },
            "optional": {
                "source_1": (anyType, {}),
                "source_2": (anyType, {}),
                "source_3": (anyType, {}),
                "source_4": (anyType, {}),
                "source_5": (anyType, {}),
                "source_6": (anyType, {}),
                "source_7": (anyType, {}),
                "source_8": (anyType, {}),
                "source_9": (anyType, {}),
                "nr_override": ("INT",{"forceInput": True}),
                #"mode_override": ("NUMBER",),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
     
    RETURN_TYPES = (anyType,"STRING",)
    RETURN_NAMES = ("SOURCE_X","Info",)

    FUNCTION = "node"
    CATEGORY = chaosaiart_higher.name("switch")
 

    def node(self, mode, nr,  
              source_0,
              source_1=None,
              source_2=None,
              source_3=None,
              source_4=None,
              source_5=None,
              source_6=None,
              source_7=None,
              source_8=None,
              source_9=None,
              #mode_override=None,
              nr_override=None):
        
        aSource = [
            source_0,
            source_1,
            source_2,
            source_3,
            source_4,
            source_5,
            source_6,
            source_7,
            source_8,
            source_9
        ]

        source_num = nr_override if nr_override is not None else nr

        mode_num = 1
        if mode == "nr = Revers":
            mode_num = 2
            aSource.reverse()
        if mode == "Random":
            mode_num = 3 
            aSource = list(filter(lambda x: x is not None, aSource))
            random_index = random.randint(0, len(aSource) - 1)
            #out_NR = random_index 
            source_num = random_index 
              
        if aSource[source_num]:
            out_NR = source_num
        else: 
            print("chaosaiart_Any_Switch: source_"+str(source_num)+" Missing")
            out_NR = None
            for i in range(source_num, len(aSource)):
                if aSource[i]:
                    out_NR = i
                    break
            if out_NR:
                print("chaosaiart_Any_Switch: Using Next Resource source_"+str(out_NR))
            else:
            
                for i in range(source_num, -1, -1):
                    if aSource[i]:
                        out_NR = i
                        break    
                
                if out_NR:
                    print("chaosaiart_Any_Switch: Using Last Resource source_"+str(out_NR))
                else:
                    print("chaosaiart_Any_Switch: Fatal error - No Resouce Found")
                    return(None,None)

     
        info = f"Used Number: {out_NR}" 
        return( aSource[out_NR], info,)
                   
 



            
 
class chaosaiart_Number:
     
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { 
                "number_int": ("INT", {"default": 1, "min": 0, "max": 99999999, "step": 1}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("INT",)
    FUNCTION = "node"
    CATEGORY = chaosaiart_higher.name("logic")

    def node(self, number_int): 
        return (number_int,)

            
 
class chaosaiart_Number2:
     
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "number_float": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10000000, "step": 0.01}), 
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("FLOAT",)
    FUNCTION = "node"
    CATEGORY = chaosaiart_higher.name("logic")

    def node(self,number_float): 
        return (number_float,)
     
 
 
class chaosaiart_controlnet_weidgth:
    @classmethod
    def INPUT_TYPES(cls):
        return { 
            "required": { 
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100000, "step": 0.01}),
                "start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100000, "step": 0.01}),
                "end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100000, "step": 0.01})
            },
            "optional": {
                "strength_override": ("FLOAT", {"forceInput": True}),
                "start_override": ("FLOAT", {"forceInput": True}),
                "end_override": ("FLOAT", {"forceInput": True}),
            }
        }
    
    RETURN_TYPES = ("FLOAT","FLOAT","FLOAT",)
    RETURN_NAMES = ("STRENGHT", "START", "END",)
    FUNCTION = "node"

    CATEGORY = chaosaiart_higher.name("controlnet")

    def node(cls,strength, start, end, strength_override=None, start_override=None, end_override=None):
        iStrength   = strength_override if not strength_override    is None     else strength
        iStart      = start_override    if not start_override       is None     else start
        iEnd        = end_override      if not end_override         is None     else end
        return (iStrength, iStart, iEnd,)
      
 
class chaosaiart_Number_Counter:
    def __init__(self): 
        #self.counters = {}
        self.counter_x = 0
        self.restartMySelf = False
        self.mode_1 = None
        self.mode_2 = None
        self.started = False 

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["Loop", "Stop_at_stop", "No_Stop"],),
                "start": ("FLOAT", {"default": 0, "min": 0, "max": 100000, "step": 0.01}),
                "stop": ("FLOAT", {"default": 1, "min": 0, "max": 100000, "step": 0.01}),
                "step": ("FLOAT", {"default": 0.1, "min": 0, "max": 100000, "step": 0.01})
            },
            "optional": {
                "restart": ("RESTART",), 
                "repeat2step": ("REPEAT",)
            },
            
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    RETURN_TYPES = ("FLOAT", "INT","STRING")
    RETURN_NAMES = ("FLOAT", "INT", "Info")

    FUNCTION = "node"
    CATEGORY = chaosaiart_higher.name("logic")
 
    def node(self, mode, start, stop, step, restart=0, repeat2step=0):
       
        start_Num = start
        stop_Num  = stop
        step_Num  = step 
        reset_Num = restart
        
        mode_1 = mode
        mode_2 = 'increment' if start_Num < stop_Num else 'decrement' 
        
        if self.restartMySelf == True:
            self.started = False 
            self.restartMySelf = False
            
        if repeat2step is None:
            repeat2step = 0

        if int(repeat2step) > 1:
            number = stop_Num - start_Num if mode_2 == 'increment' else start_Num - stop_Num
            step_Num = round(number / (repeat2step-1),3)

            #step_Num = number / repeat2step
            #print("Chaosaiart - Number Counter: steps_override, Steps = "+str(step_Num))
             
        if not int(repeat2step) == 1:
            if (self.mode_1 == mode_1 and self.mode_2 == mode_2):
                if self.started == True:
                    if not reset_Num >= 1:     
                        if not stop_Num == start_Num:
                            counter = self.counter_x
                            counter = counter + step_Num if mode_2 == 'increment' else counter - step_Num
                            counter = round(counter, 3)

                            FloatBugFix = 0.005
                            if counter >= stop_Num - FloatBugFix:  
                                if mode_2 == 'increment': 
                                    counter = stop_Num   
                                    if mode_1 == "Loop": 
                                        self.restartMySelf = True 

                            if counter <= stop_Num + FloatBugFix: 
                                if not mode_2 == 'increment': 
                                    counter = stop_Num
                                    if mode_1 == "Loop": 
                                        self.restartMySelf = True
        
                            counter = counter if counter >= 0 else 0

                        else:
                            counter = start_Num
                    else:
                        self.restartMySelf = False
                        counter = start_Num
                        print("Chaosaiart - Number Counter: restarted")
                else:
                    self.started = True 
                    self.restartMySelf = False
                    counter = start_Num
                    print("Chaosaiart - Number Counter: Started")
            else:
                self.started = True 
                self.restartMySelf = False
                counter = start_Num
                print("Chaosaiart - Number Counter: Mode Change and restarted")
            
            self.mode_1 = mode_1
            self.mode_2 = mode_2
            self.counter_x = counter

            if step_Num != 0:
                info = "chaosaiart - NumberCounter: No Stop\n"
                if not mode_1 == "No_Stop":
                    if mode_1 == "Loop":
                        info = "Loop repeat in: \n"
                    else:
                        info = "End in: \n"
                    info +=  str(int(abs((stop_Num - self.counter_x) / step_Num))) 
            else:
                info= "no Count.\n"    
        else:
            counter = start_Num
            self.counter_x = counter
            info= "Inpute repeat 1 counter = start.\n"   
            

        info = f"Output: {counter}\n"+info   
        return ( float(counter), int(counter), info )    

 
class chaosaiart_Simple_Prompt: 
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Prompt": ("STRING", {"multiline": True}), 
            },
            "optional":{ 
                "add_prompt": ("STRING", {"multiline": True, "forceInput": True}),
            } 
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("PROMPT_TXT",)

    FUNCTION = "node"

    CATEGORY = chaosaiart_higher.name("prompt")
 
    def node(self, Prompt="", add_prompt=""): 
        out = chaosaiart_higher.add_Prompt_txt__replace_randomPart(add_prompt,Prompt)
        return(out,)

"""    
#NOTIZ: Add_prompt / Add_positv .. kick it out, no Reason for it anymore    
class chaosaiart_ADD_Prompt:
    location = ["after","before"]  
    #separator = ["comma", "space", "newline", "none"]
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "addition": ("STRING", {"multiline": True}),
                "placement": (s.location,),
                #"separator": (s.separator,),
            },
            "optional": {
                "input_string": ("STRING", {"multiline": True, "forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("TEXT",)

    FUNCTION = "node"

    CATEGORY = chaosaiart_higher.name("prompt")
 
    def node(self,placement, addition="",  input_string=""):

        if (input_string is None):
            return(addition,)

        if (placement == "after"):
            new_string = input_string + "," + addition
        else:
            new_string = addition + "," + input_string

        return(new_string,)
"""

#TODO: Testen, after changing
class chaosaiart_Prompt_Frame: 
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "start_frame": ("INT",{"default": 1, "min": 1, "max": 18446744073709551615, "step": 1}),
                "positiv": ("STRING", {"multiline": True}),
                "negativ": ("STRING", {"multiline": True}),
            },
            "optional": {
                "add_positiv": ("STRING", {"multiline": True, "forceInput": True}),
                "add_negativ": ("STRING", {"multiline": True, "forceInput": True}),
                "add_lora": ("LORA",),
            },
        }

    RETURN_TYPES = ("FRAME_PROMPT",)
    RETURN_NAMES = ("FRAME_PROMPT",)

    FUNCTION = "node"

    CATEGORY = chaosaiart_higher.name("prompt")
 
    def node(self,start_frame,positiv="",negativ="",add_lora=[],add_positiv="",add_negativ=""):
        
        positivOUT = chaosaiart_higher.add_Prompt_txt__replace_randomPart(add_positiv,positiv)
        negativOUT = chaosaiart_higher.add_Prompt_txt__replace_randomPart(add_negativ,negativ)
        return([start_frame,[positivOUT,negativOUT,add_lora]],)
    
class chaosaiart_convert_Prompt:
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return { 
            "required": { 
            },
            "optional": {
                "add_frame_prompt":("FRAME_PROMPT",), 
                "add_positiv": ("STRING", {"multiline": True, "forceInput": True}),
                "add_negativ": ("STRING", {"multiline": True, "forceInput": True}),
                "add_lora": ("LORA",),
            },
        }

    RETURN_TYPES = ("MAIN_PROMPT","STRING")
    RETURN_NAMES = ("MAIN_PROMPT","Info")

    FUNCTION = "node"

    CATEGORY = chaosaiart_higher.name("prompt")

    def node(self,add_frame_prompt = None,add_lora=[],add_positiv="",add_negativ=""):
        frame_prompt_postiv         = ""
        frame_prompt_negativ        = ""
        frame_prompt_lora           = []

        if add_frame_prompt is not None: 
            frame_prompt_postiv     = add_frame_prompt[1][0]
            frame_prompt_negativ    = add_frame_prompt[1][1]
            frame_prompt_lora       = add_frame_prompt[1][2]

        add_lora += frame_prompt_lora
        positivOUT = chaosaiart_higher.add_Prompt_txt__replace_randomPart(add_positiv,frame_prompt_postiv)
        negativOUT = chaosaiart_higher.add_Prompt_txt__replace_randomPart(add_negativ,frame_prompt_negativ)

        info = f"Positiv:\n{positivOUT}\n\n"
        info += f"negativOUT:\n{negativOUT}\n\n"
        info += f"Lora:\n" + ", ".join(add_lora)
         
        return [positivOUT,negativOUT,add_lora],info,

class chaosaiart_Prompt: 
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positiv": ("STRING", {"multiline": True}),
                "negativ": ("STRING", {"multiline": True}),
            },
            "optional": {
                "add_positiv": ("STRING", {"multiline": True, "forceInput": True}),
                "add_negativ": ("STRING", {"multiline": True, "forceInput": True}),
                "add_lora": ("LORA",),
            },
        }

    RETURN_TYPES = ("MAIN_PROMPT","LORA","STRING","STRING",)
    RETURN_NAMES = ("MAIN_PROMPT","LORA","POSITIV_TXT","NEGATIV_TXT",)

    FUNCTION = "node"

    CATEGORY = chaosaiart_higher.name("prompt")
    
    def node(self,positiv="",negativ="",add_lora=[],add_positiv="",add_negativ=""):
        positivOUT = chaosaiart_higher.add_Prompt_txt__replace_randomPart(add_positiv,positiv)
        negativOUT = chaosaiart_higher.add_Prompt_txt__replace_randomPart(add_negativ,negativ)
        return([positivOUT,negativOUT,add_lora],add_lora,positivOUT,negativOUT,)
        

class chaosaiart_Prompt_mixer_byFrame: 
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "activ_frame": ("ACTIV_FRAME", ),
            },
            "optional": {   
                "main_prompt": ("MAIN_PROMPT",),
                "frame_prompt_1": ("FRAME_PROMPT",),
                "frame_prompt_2": ("FRAME_PROMPT",),
                "frame_prompt_3": ("FRAME_PROMPT",),
                "frame_prompt_4": ("FRAME_PROMPT",),
                "frame_prompt_5": ("FRAME_PROMPT",),
                "frame_prompt_6": ("FRAME_PROMPT",),
                "frame_prompt_7": ("FRAME_PROMPT",),
                "frame_prompt_8": ("FRAME_PROMPT",),
                "frame_prompt_9": ("FRAME_PROMPT",),
            }
        }

    RETURN_TYPES = ("STRING","FRAME_PROMPT","LORA","STRING","STRING",)
    RETURN_NAMES = ("Info","FRAME_PROMPT","LORA","POSITIV_TXT","NEGAITV_TXT",)

    FUNCTION = "node"

    CATEGORY = chaosaiart_higher.name("prompt")

    def node(self,activ_frame=0,
                 main_prompt=None,
                 frame_prompt_1=None,
                 frame_prompt_2=None,
                 frame_prompt_3=None,
                 frame_prompt_4=None,
                 frame_prompt_5=None,
                 frame_prompt_6=None,
                 frame_prompt_7=None,
                 frame_prompt_8=None,
                 frame_prompt_9=None):
          
        Frame_array = [
            frame_prompt_9,
            frame_prompt_8,
            frame_prompt_7,
            frame_prompt_6,
            frame_prompt_5,
            frame_prompt_4,
            frame_prompt_3,
            frame_prompt_2,
            frame_prompt_1
            ]

        main_positiv    = ""
        main_negativ    = "" 
        main_Lora       = []
        frame_Lora      = []
        frame_positiv   = ""
        frame_negativ   = ""
                    
        if main_prompt:
            main_positiv += str(main_prompt[0])
            main_negativ += str(main_prompt[1])
            main_Lora = main_prompt[2]

        frame_key = 0
        Frame_sorted_inputs = sorted(filter(None, Frame_array), key=lambda x: x[0], reverse=True)

        for array in Frame_sorted_inputs:
            if isinstance(array, list): 
                if activ_frame >= int(array[0]):

                    frame_positiv  = str(array[1][0])
                    frame_negativ  = str(array[1][1])
                    frame_Lora = array[1][2]
                    frame_key = int(array[0])
                    break
        
        positiv = chaosaiart_higher.add_Prompt_txt__replace_randomPart(main_positiv,frame_positiv)
        negativ = chaosaiart_higher.add_Prompt_txt__replace_randomPart(frame_negativ,main_negativ)
        lora =  chaosaiart_higher.lora_mainprompt_and_frame(main_Lora,frame_Lora)
        
        info =  f"Frame: {activ_frame},\nPositiv: {positiv},\nNegativ: {negativ}\n"
        info += chaosaiart_higher.lora_info(lora)
        return(info,[frame_key,[positiv,negativ,lora]],lora,positiv,negativ,)

 

"""
#NOTIZ: Kicked out by Add_prompt function
class chaosaiart_Prompt_mixer: 
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "prompt_txt_1": ("STRING", {"multiline": True, "forceInput": True}),
            },
            "optional": {     
                "prompt_txt_2": ("STRING", {"multiline": True, "forceInput": True}),
                "prompt_txt_3": ("STRING", {"multiline": True, "forceInput": True}),
                "prompt_txt_4": ("STRING", {"multiline": True, "forceInput": True}),
                "prompt_txt_5": ("STRING", {"multiline": True, "forceInput": True}),
                "prompt_txt_6": ("STRING", {"multiline": True, "forceInput": True}),
                "prompt_txt_7": ("STRING", {"multiline": True, "forceInput": True}),
                "prompt_txt_8": ("STRING", {"multiline": True, "forceInput": True}),
                "prompt_txt_9": ("STRING", {"multiline": True, "forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING","STRING",)
    RETURN_NAMES = ("Info","PROMPT_TXT",)

    FUNCTION = "node"

    CATEGORY = chaosaiart_higher.name("prompt")

    def node(self,
                 prompt_txt_1=None,
                 prompt_txt_2=None,
                 prompt_txt_3=None,
                 prompt_txt_4=None,
                 prompt_txt_5=None,
                 prompt_txt_6=None,
                 prompt_txt_7=None,
                 prompt_txt_8=None,
                 prompt_txt_9=None):
          
        out = ""
        out =            prompt_txt_1 if prompt_txt_1 else out
        out = out +", "+ prompt_txt_2 if prompt_txt_2 else out
        out = out +", "+ prompt_txt_3 if prompt_txt_3 else out
        out = out +", "+ prompt_txt_4 if prompt_txt_4 else out
        out = out +", "+ prompt_txt_5 if prompt_txt_5 else out
        out = out +", "+ prompt_txt_6 if prompt_txt_6 else out
        out = out +", "+ prompt_txt_7 if prompt_txt_7 else out
        out = out +", "+ prompt_txt_8 if prompt_txt_8 else out
        out = out +", "+ prompt_txt_9 if prompt_txt_9 else out

     
        return(out,out,)
"""

 
imgType_EXT = ["jpg", "jpeg", "png"]

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def file_by_directory(directory_path,AllowedType):

    Array = []
    for root, _, items in os.walk(directory_path):
        for item in items:
            if item.lower().endswith(tuple(AllowedType)):
                Array.append(os.path.join(root, item))

    return Array           
   
class chaosaiart_merge_Folders:                
    def __init__(self):    
        self.output_dir = folder_paths.get_output_directory() 

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {  
                "Output_Folder": ("STRING", {"default": 'merge_folders_chaosaiart', "multiline": False}), 
                #"if_Size_not_fit":(["resize","crop","fill"],), 
                "if_Size_not_fit":(["crop","resize"],), 
                "path_1": ("STRING", {"default": '', "multiline": False}), 
            },
            "optional": {   
                "path_2": ("STRING", {"default": '', "multiline": False}),   
                "path_3": ("STRING", {"default": '', "multiline": False}),   
                "path_4": ("STRING", {"default": '', "multiline": False}),   
                "path_5": ("STRING", {"default": '', "multiline": False}),   
                "path_6": ("STRING", {"default": '', "multiline": False}),   
                "path_7": ("STRING", {"default": '', "multiline": False}),   
                "path_8": ("STRING", {"default": '', "multiline": False}),   
                "path_9": ("STRING", {"default": '', "multiline": False}),   
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")


    RETURN_TYPES = ("STRING","PATH",)
    RETURN_NAMES = ("Info","MERGE_FOLDERS",)
    FUNCTION = "node"

    OUTPUT_NODE = True

    CATEGORY = chaosaiart_higher.name("video")

    def node(self, if_Size_not_fit, Output_Folder, path_1,
                path_2=None,
                path_3=None,
                path_4=None,
                path_5=None,
                path_6=None,
                path_7=None,
                path_8=None,
                path_9=None):

        num = 0
        while os.path.exists(os.path.join(self.output_dir, Output_Folder, f"v_{num:04d}")):
            num += 1

        output_dir = os.path.join(self.output_dir, Output_Folder, f"v_{num:04d}")

        paths = [path_1, path_2, path_3, path_4, path_5, path_6, path_7, path_8, path_9]
        info = f"Output dir: {output_dir}\nUsed Folders:\n"
        info_list_NoCopy = ""
        infoIMG = ""
        
        width, height, img_type = None,None,None
        
        for index, path in enumerate(paths):
            if path is not None:
                if os.path.exists(path):
                    info += f"Path_{index + 1} Used.\n"
                    files = os.listdir(path)   
                    
                    files_progress = tqdm(files, desc=f"Processing Path_{index + 1}", unit="file")
                    for i, file in enumerate(files_progress):
                    #for i, file in enumerate(files):
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            image_path = os.path.join(path, file)
                            image_type = os.path.splitext(file)[1]  
                            image = Image.open(image_path)
                            
                            if width is None or height is None or img_type is None: 
                                width = image.size[0]
                                height = image.size[1]
                                img_type = image_type
                                infoIMG = f"Cover all IMG into {img_type} - {width}x{height}"
                                 
                                if not os.path.exists(output_dir):
                                    os.makedirs(output_dir)
                            else:
                                if not img_type == image_type: 
                                    image_type = img_type

                                m = {"resize":"resize","crop":"crop","fill":"fill"}
                                resize_type = m.get(if_Size_not_fit) 
                                image = chaosaiart_higher.resize_image(resize_type, image, width, height)
                                    
                            new_filename = f"img_p{index + 1}_{i:06d}{image_type}" 
                            path_activ = os.path.join(output_dir, new_filename)   
                            image.save(path_activ)
                        else:
                            info_list_NoCopy += f"{file}\n"
                else:
                    info += f"Path_{index + 1} not found.\n"

        info = infoIMG + info + f"Not Copyed:\n{info_list_NoCopy}"
        return info, output_dir

 
class chaosaiart_Load_Image_Batch:
    def __init__(self):   
        self.counter = 0 
        self.activ_index = 0
        self.path = ""
        self.Started = False
 
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { 
                "index": ("INT", {"default": 1, "min": 1, "max": 150000, "step": 1}), 
                "path": ("STRING", {"default": '', "multiline": False}),  
            },
            "optional": { 
                "restart": ("RESTART",),
                "activ_frame2index": ("ACTIV_FRAME",),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")


    RETURN_TYPES = ("STRING","IMAGE",)
    RETURN_NAMES = ("Info","IMAGE",)
    FUNCTION = "node"

    CATEGORY = chaosaiart_higher.name("image")

    def node(self, path, restart=0, activ_frame2index=None, index=1):
        
        indexNum = activ_frame2index if activ_frame2index is not None else index
        indexNum = indexNum - 1  
         
        if not self.path == path:
            self.path = path
            self.Started = False

        if restart > 0 or self.Started == False:
            self.counter = 0
 
        self.Started = True

        if os.path.exists(path):
            
            if activ_frame2index is None:
                indexNum = indexNum + self.counter 
                self.counter += 1 
                chaosaiart_higher.Debugger("chaosaiart_Load_Image_Batch",f"no activ_frame2index + Index: {indexNum}")

            fileArray = file_by_directory(path,imgType_EXT)
            if len(fileArray) >= 1:
                
                if indexNum < 0:
                    indexNum = 0
                    print("Chaosaiart - Load Image Batch: Below first image not allowed, repeat first image")

                if indexNum >= len(fileArray):
                    indexNum = len(fileArray)-1 
                    print("Chaosaiart - Load Image Batch: its Last IMG, i will repeat it.")
 
                image = Image.open(fileArray[indexNum]) 
                self.activ_index = indexNum

            else:
                self.Started = False 
                info = "Chaosaiart - Load Batch Image: Error: Empty Directory!"
                print("----")
                print(" ")
                print("\033[91m"+info+"\033[0m")
                print(" ")
                print("----")
                return info,None,
               
        else:
            self.Started = False
            info = "Chaosaiart - Load Image Batch: Error: Open Directory Path Failed" 
            print("----")
            print(" ")
            print("\033[91m"+info+"\033[0m")
            print(" ")
            print("----")
            return info,None, 
            
        info = f"Index: {self.activ_index+1}"  
        return (info, pil2tensor(image),) 
  
 
class chaosaiart_Load_Image_Batch_2img:
    def __init__(self): 
        self.image_history = None 
        self.image_history2 = None 
        self.repeatCount = 0   
        self.counter = 0
        self.Started = False
        self.activ_index = 0
        self.path = ""
 
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { 
                "index": ("INT", {"default": 1, "min": 1, "max": 150000, "step": 1}), 
                "path": ("STRING", {"default": '', "multiline": False}), 
                "repeat": ("INT", {"default": 0, "min": 0, "max": 150000, "step": 1}), 
            },
            "optional": { 
                "restart": ("RESTART",),
                "repeat_Override": ("REPEAT",),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")


    RETURN_TYPES = ("STRING", "IMAGE","IMAGE","REPEAT",)
    RETURN_NAMES = ("Info","IMAGE = Index","IMAGE = Index+1","REPEAT",)
    FUNCTION = "node"

    CATEGORY = chaosaiart_higher.name("image")

    def node(self, path,restart=0, repeat_Override=None, repeat=0, index=0):
      
        newImageCheck = False
        
        indexNum    = index 
        indexNum    = indexNum - 1
        repeatNum   = repeat_Override     if repeat_Override is not None else repeat 
        restartNum  = restart
         
        if not self.path == path:
            self.path = path
            self.Started == False

        if restartNum > 0 or self.Started == False:
            self.image_history = None 
            self.repeatCount = 0 
            self.counter = 0
 

        if  self.repeatCount >= repeatNum:  
            self.repeatCount = 0

        if self.repeatCount == 0:
            self.Started = True
            newImageCheck = True 

            if os.path.exists(path):
  
                indexNum = indexNum + self.counter 
                indexNum2 = indexNum + 1
                self.counter += 1 

                fileArray = file_by_directory(path,imgType_EXT)
                if len(fileArray) >= 1:

                    if indexNum < 0:
                        indexNum = 0
                        indexNum2 = 1
                        print("Chaosaiart - Load Image Batch: Below first image not allowed, repeat first image")

                    if indexNum >= len(fileArray):
                        indexNum = len(fileArray)-1 
                        print("Chaosaiart - Load Image Batch: its Last IMG, i will repeat it.")
                    if indexNum2 >= len(fileArray):
                        indexNum2 = len(fileArray)-1 
                        print("Chaosaiart - Load Image Batch: its Last IMG, i will repeat it.")
                
                    image = Image.open(fileArray[indexNum])
                    image2 = Image.open(fileArray[indexNum2])
                    self.activ_index = indexNum

                else:
                    self.Started = False
                    info = "Chaosaiart - Load Batch Image: Error: Empty Directory!"
                    print("----")
                    print(" ")
                    print("\033[91m"+info+"\033[0m")
                    print(" ")
                    print("----")
                    return info,None,None,None,
               
            else:
                self.Started = False
                info = "Chaosaiart - Load Image Batch: Error: Open Directory Path Failed" 
                print("----")
                print(" ")
                print("\033[91m"+info+"\033[0m")
                print(" ")
                print("----")
                return info,None,None,None,
                
             
            self.image_history = image 
            self.image_history2 = image2  

             
        repeat_OUT = repeatNum - self.repeatCount 
        self.repeatCount += 1
        info = f"Index: {self.activ_index+1}\nCountdown: {repeat_OUT}" 

        if newImageCheck: 
            return ( info, pil2tensor(image), pil2tensor(image2), repeatNum, )
        return ( info, pil2tensor(self.image_history), pil2tensor(self.image_history2), repeatNum, )
  

  
class chaosaiart_CheckpointLoader:
    def __init__(self):  

        self.Cache_index    = None
        self.Cache_CKPT     = None
        self.Cache_Lora     = None
        self.Cache_pPrompt  = None
        self.Cache_nPrompt  = None 

        self.lora_load_cache = []

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "activ_frame":("ACTIV_FRAME",),
                "positiv_txt": ("STRING", {"multiline": True, "forceInput": True}),
                "negativ_txt": ("STRING", {"multiline": True, "forceInput": True}),
         
                "ckpt_1": (folder_paths.get_filename_list("checkpoints"), ),
                "ckpt_1_info": (["ckpt_1_Start will start",], ),
                
            },
            "optional": { 
                
                "ckpt_2": (folder_paths.get_filename_list("checkpoints"), ),
                "ckpt_2_Frame_start": ("INT", {"default": 0, "min": 0, "max": 150000, "step": 1}),
                
                "ckpt_3": (folder_paths.get_filename_list("checkpoints"), ),
                "ckpt_3_Frame_start": ("INT", {"default": 0, "min": 0, "max": 150000, "step": 1}), 
                
                "ckpt_4": (folder_paths.get_filename_list("checkpoints"), ),
                "ckpt_4_Frame_start": ("INT", {"default": 0, "min": 0, "max": 150000, "step": 1}), 

                "ckpt_5": (folder_paths.get_filename_list("checkpoints"), ),
                "ckpt_5_Frame_start": ("INT", {"default": 0, "min": 0, "max": 150000, "step": 1}),
                
                "ckpt_6": (folder_paths.get_filename_list("checkpoints"), ),
                "ckpt_6_Frame_start": ("INT", {"default": 0, "min": 0, "max": 150000, "step": 1}),
                
                "ckpt_7": (folder_paths.get_filename_list("checkpoints"), ),
                "ckpt_7_Frame_start": ("INT", {"default": 0, "min": 0, "max": 150000, "step": 1}),
                
                "ckpt_8": (folder_paths.get_filename_list("checkpoints"), ),
                "ckpt_8_Frame_start": ("INT", {"default": 0, "min": 0, "max": 150000, "step": 1}),
                
                "ckpt_9": (folder_paths.get_filename_list("checkpoints"), ),
                "ckpt_9_Frame_start": ("INT", {"default": 0, "min": 0, "max": 150000, "step": 1}),
                
                "lora":("LORA",)
            }
        }
    
    
    #RETURN_TYPES = ("STRING","MODEL", "CLIP", "VAE",)
    #RETURN_NAMES = ("Info","MODEL", "CLIP", "VAE",)
    RETURN_TYPES = ("STRING","MODEL", "CONDITIONING","CONDITIONING", "VAE",)
    RETURN_NAMES = ("Info","MODEL", "POSITIV","NEGATIV", "VAE",)
    FUNCTION = "node"

    CATEGORY = chaosaiart_higher.name("checkpoint")

    def node(self, 
                        ckpt_1,positiv_txt="", negativ_txt="",
                        ckpt_2=None,
                        ckpt_3=None,
                        ckpt_4=None,
                        ckpt_5=None,
                        ckpt_6=None,
                        ckpt_7=None,
                        ckpt_8=None,
                        ckpt_9=None,
                        activ_frame=0,
                        ckpt_1_info=None,
                        ckpt_2_Frame_start=0,
                        ckpt_3_Frame_start=0,
                        ckpt_4_Frame_start=0,
                        ckpt_5_Frame_start=0,
                        ckpt_6_Frame_start=0,
                        ckpt_7_Frame_start=0,
                        ckpt_8_Frame_start=0,
                        ckpt_9_Frame_start=0,
                        lora = [],output_vae=True, output_clip=True):
        
         
        ckpt_name_ByFrame = ckpt_1  

        if activ_frame > 1:
            Frame_array = [
                [0,ckpt_1],
                [ckpt_2_Frame_start,ckpt_2], 
                [ckpt_3_Frame_start,ckpt_3], 
                [ckpt_4_Frame_start,ckpt_4], 
                [ckpt_5_Frame_start,ckpt_5], 
                [ckpt_6_Frame_start,ckpt_6], 
                [ckpt_7_Frame_start,ckpt_7], 
                [ckpt_8_Frame_start,ckpt_8], 
                [ckpt_9_Frame_start,ckpt_9] 
            ]
    
            Frame_array = [item for item in Frame_array if item[1] is not None]
            Frame_array_filtered = [Frame_array[0]] + [item for item in Frame_array[1:] if item[0] != 0]
            Frame_array_filtered.sort(key=lambda x: x[0], reverse=True)

            for array in Frame_array_filtered:
                if isinstance(array, list): 
                    if activ_frame >= int(array[0]):
                        ckpt_name_ByFrame = array[1]
                        break 
          
        ckpt_name   = ckpt_name_ByFrame
        alora       = lora
        sPositiv    = positiv_txt
        sNegativ    = negativ_txt

        self.Cache_index, bCKPT_inCache, bLora_inCache, bPositiv_inCache ,bNegativ_inCache \
        = chaosaiart_higher.check_Checkpoint_Lora_Txt_caches(self.Cache_index,ckpt_name,alora,sPositiv,sNegativ)

        if not bCKPT_inCache:
            self.Cache_CKPT = chaosaiart_higher.checkpointLoader(ckpt_name) 
        MODEL, CLIP, VAE = self.Cache_CKPT    

        if not ( bCKPT_inCache and bLora_inCache):    
            self.Cache_Lora = chaosaiart_higher.load_lora_by_Array(alora, MODEL, CLIP, self.lora_load_cache)
        MODEL, positiv_clip, negativ_clip, self.lora_load_cache, lora_Info = self.Cache_Lora

        if not ( bCKPT_inCache and bLora_inCache and bPositiv_inCache):   
            self.Cache_pPrompt = chaosaiart_higher.textClipEncode(positiv_clip, sPositiv) 
        PositivOut = self.Cache_pPrompt
        
        if not ( bCKPT_inCache and bLora_inCache and bNegativ_inCache):
            self.Cache_nPrompt = chaosaiart_higher.textClipEncode(negativ_clip, sNegativ) 
        NegativOut = self.Cache_nPrompt   
        
        info = f"Frame: {activ_frame}\nCheckpoint: {ckpt_name}\nPositiv: {positiv_txt}\nNegativ: {negativ_txt}\n{lora_Info}"
        return (info,MODEL,PositivOut,NegativOut,VAE,) 
   
 




    
class chaosaiart_convert:

    @classmethod
    def INPUT_TYPES(cls):
        return { 
            "required": { 
                "source": (anyType, {})
            }
        }
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN") 
    
    RETURN_TYPES = ("FLOAT","INT","NUMBER","ACTIV_FRAME","REPEAT","RESTART","Bool","STRING",)
    RETURN_NAMES = ("FLOAT", "INT", "NUMBER","ACTIV_FRAME","REPEAT","RESTART(reset=1)","Bool","STRING",)
    FUNCTION = "node"

    CATEGORY = chaosaiart_higher.name("logic")

    def node(cls,source):

        strSource   = str(source)
        intSource   = None
        floatSource = None
        bool = False

        if chaosaiart_higher.is_number(source):
            intSource = int(source)
            floatSource = float(source)
            
        if source == 1 or source == "1" or strSource.lower() == "true":
            bool = True 

        return floatSource, intSource, floatSource, intSource,intSource,intSource, bool, strSource,
      

class chaosaiart_oneNode:
    def __init__(self):  
        self.image_history  = None
        self.latent_history = None
        self.notStarted     = True

        self.Cache_index    = None
        self.Cache_CKPT     = None
        self.Cache_Lora     = None
        self.Cache_pPrompt  = None
        self.Cache_nPrompt  = None 

        self.lora_load_cache = []
        self.device = comfy.model_management.intermediate_device() 
  
      
    @classmethod
    def INPUT_TYPES(s):
        return {    
            "required":{ 
                    "Mode": (["Create Image","Create Image, Hold it until Restart"],), 
                    "Checkpoint": (folder_paths.get_filename_list("checkpoints"), ),
                    "Positiv": ("STRING", {"multiline": True}),
                    "Negativ": ("STRING", {"multiline": True}), 
                    "empty_Img_width": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                    "empty_Img_height": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}), 
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "denoise": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ), 
                },
                "optional":{
                    "restart": ("RESTART", ),
                    "add_main_prompt": ("MAIN_PROMPT",), 
                    "latent_override": ("LATENT",),
                    "img2img_override": ("IMAGE",),
                    "vae_override": ("VAE", ),
                }
            }


    RETURN_TYPES = ("IMAGE","LATENT",)
    RETURN_NAMES = ("IMAGE","LATENT",)
    FUNCTION = "node"

    CATEGORY = chaosaiart_higher.name("ksampler")

    @staticmethod
    def vae_encode_crop_pixels(pixels):
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
        return pixels

    def node(self,  Mode, Checkpoint,
            empty_Img_width, empty_Img_height, seed, steps, cfg, sampler_name, scheduler, denoise,
            add_main_prompt = None,
            Positiv = "", Negativ = "", latent_override = None, img2img_override = None,  
            vae_override = None, restart = 0
        ):
        
        if Mode == "Create Image" or restart >= 1 or self.notStarted:

            add_positiv_txt = ""    if add_main_prompt is None else add_main_prompt[0] 
            add_negativ_txt = ""    if add_main_prompt is None else add_main_prompt[1] 
            add_lora = []           if add_main_prompt is None else add_main_prompt[2] 
             
            self.notStarted = False
            ckpt_name       = Checkpoint
            alora           = add_lora
            sPositiv        = chaosaiart_higher.add_Prompt_txt__replace_randomPart(add_positiv_txt,Positiv)
            sNegativ        = chaosaiart_higher.add_Prompt_txt__replace_randomPart(Negativ,add_negativ_txt)  

            self.Cache_index, bCKPT_inCache, bLora_inCache, bPositiv_inCache ,bNegativ_inCache \
            = chaosaiart_higher.check_Checkpoint_Lora_Txt_caches(self.Cache_index,ckpt_name,alora,sPositiv,sNegativ)

            if not bCKPT_inCache:
                self.Cache_CKPT = chaosaiart_higher.checkpointLoader(ckpt_name) 
            MODEL, CLIP, VAE = self.Cache_CKPT    

            if not ( bCKPT_inCache and bLora_inCache):    
                self.Cache_Lora = chaosaiart_higher.load_lora_by_Array(alora, MODEL, CLIP, self.lora_load_cache)
            MODEL, positiv_clip, negativ_clip, self.lora_load_cache, lora_Info = self.Cache_Lora

            if not ( bCKPT_inCache and bLora_inCache and bPositiv_inCache):   
                self.Cache_pPrompt = chaosaiart_higher.textClipEncode(positiv_clip, sPositiv) 
            PositivOut = self.Cache_pPrompt
            
            if not ( bCKPT_inCache and bLora_inCache and bNegativ_inCache):
                self.Cache_nPrompt = chaosaiart_higher.textClipEncode(negativ_clip, sNegativ) 
            NegativOut = self.Cache_nPrompt   


            #def node(self, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise,empty_Img_width,empty_Img_height,empty_Img_batch_size,latent_Override=None,latent_by_Image_Override=None,denoise_Override=None):
            model = MODEL
            vae = VAE if vae_override is None else vae_override
            positive = PositivOut
            negative = NegativOut
            empty_Img_batch_size = 1

            if img2img_override is None: 
                if latent_override is None:
                    latent = torch.zeros([empty_Img_batch_size, 4, empty_Img_height // 8, empty_Img_width // 8], device=self.device)
                    latent_image = {"samples":latent}
                    if not denoise==1:
                        print("chaosaiart_oneNode: set Denoising to 1")
                    denoise = 1 
                else: 
                    latent_image = latent_override
            else:
                pixels = img2img_override
                pixels = self.vae_encode_crop_pixels(pixels)
                t = vae.encode(pixels[:,:,:,:3])
                latent_image = {"samples":t} 

            samples = chaosaiart_higher.ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)
            image = vae.decode(samples[0]["samples"])
            self.image_history = image 
            self.latent_history = samples[0]

        return self.image_history , self.latent_history,
    
class chaosaiart_ControlNetApply:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "control_net": ("CONTROL_NET", ),
                             "image": ("IMAGE", ),
                             "strength": ("FLOAT", {"forceInput": True}),
                             "start": ("FLOAT", {"forceInput": True}),
                             "end": ("FLOAT", {"forceInput": True})
                        
                             }}

    RETURN_TYPES = ("CONDITIONING","CONDITIONING")
    RETURN_NAMES = ("POSITVE", "NEGATIVE")
    FUNCTION = "node"

    CATEGORY = chaosaiart_higher.name("controlnet")

    def node(self, positive, negative, control_net, image, strength, start, end):
        if strength == 0:
            return (positive, negative) 
        
        if start == end:
            return (positive, negative)
        
        control_hint = image.movedim(-1,1)
        cnets = {}

        out = []
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()

                prev_cnet = d.get('control', None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    c_net = control_net.copy().set_cond_hint(control_hint, strength, (start, end))
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net

                d['control'] = c_net
                d['control_apply_to_uncond'] = False
                n = [t[0], d]
                c.append(n)
            out.append(c)
        return (out[0], out[1])


class chaosaiart_ControlNetApply2:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "control_net": ("CONTROL_NET", ),
                             "image": ("IMAGE", ),
                             "strength": ("FLOAT",  {"default": 1, "min": 0, "max": 3, "step": 0.01}),
                             "start": ("FLOAT",  {"default": 0, "min": 0, "max": 1, "step": 0.01}),
                             "end": ("FLOAT",  {"default": 1, "min": 0, "max": 1, "step": 0.01})
                             }}

    RETURN_TYPES = ("CONDITIONING","CONDITIONING")
    RETURN_NAMES = ("POSITVE", "NEGATIVE")
    FUNCTION = "node"

    CATEGORY = chaosaiart_higher.name("controlnet")

    def node(self, positive, negative, control_net, image, strength, start, end):
        if strength == 0:
            return (positive, negative)
        if start == end:
            return (positive, negative)    

        control_hint = image.movedim(-1,1)
        cnets = {}

        out = []
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()

                prev_cnet = d.get('control', None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    c_net = control_net.copy().set_cond_hint(control_hint, strength, (start, end))
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net

                d['control'] = c_net
                d['control_apply_to_uncond'] = False
                n = [t[0], d]
                c.append(n)
            out.append(c)
        return (out[0], out[1])
    


    
class chaosaiart_ControlNetApply3:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "activ_frame": ("ACTIV_FRAME",),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "control_net": ("CONTROL_NET", ),
                    "image": ("IMAGE", ),
                    "strength_override":("FLOAT",{"forceInput": True}),
                    "start_override":("FLOAT",{"forceInput": True}),
                    "end_override":("FLOAT",{"forceInput": True}),
                    "strength": ("FLOAT",  {"default": 1, "min": 0, "max": 3, "step": 0.01}),
                    "start": ("FLOAT",  {"default": 0, "min": 0, "max": 1, "step": 0.01}),
                    "end": ("FLOAT",  {"default": 1, "min": 0, "max": 1, "step": 0.01}),
                    "start_Frame":("INT", {"default": 1, "min": 1, "max": 999999999, "step": 1}),
                    "End_Frame":("INT", {"default": 9999, "min": 1, "max": 999999999, "step": 1}),
                }}

    RETURN_TYPES = ("CONDITIONING","CONDITIONING")
    RETURN_NAMES = ("POSITVE", "NEGATIVE")
    FUNCTION = "node"

    CATEGORY = chaosaiart_higher.name("controlnet")

    def node(self, positive, negative, control_net, image, strength, start, end, start_Frame, End_Frame,activ_frame, strength_override= None,start_override = None,end_override=None):
        if not ( activ_frame >= start_Frame and  activ_frame < End_Frame ):    
            return (positive, negative)
        
        strength    = strength  if strength_override    is None else strength_override
        start       = start     if start_override       is None else start_override
        end         = end       if end_override         is None else end_override
 
        if strength == 0:
            return (positive, negative)
        if start == end:
            return (positive, negative)    

        control_hint = image.movedim(-1,1)
        cnets = {}

        out = []
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()

                prev_cnet = d.get('control', None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    c_net = control_net.copy().set_cond_hint(control_hint, strength, (start, end))
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net

                d['control'] = c_net
                d['control_apply_to_uncond'] = False
                n = [t[0], d]
                c.append(n)
            out.append(c)
        return (out[0], out[1])





class chaosaiart_adjust_color:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "image":("IMAGE",),     
                        "Contrast": ("FLOAT", {"default": 1, "min": 0.01, "max": 2, "step": 0.01}),
                        "Color": ("FLOAT", {"default": 1, "min": 0, "max": 5, "step": 0.01}),
                        "Brightness": ("FLOAT", {"default": 1, "min": 0.2, "max": 2, "step": 0.01}),
                        "Red": ("FLOAT", {"default": 1, "min": 0, "max": 5, "step": 0.01}),
                        "Green": ("FLOAT", {"default": 1, "min": 0, "max": 5, "step": 0.01}),
                        "Blue": ("FLOAT", {"default": 1, "min": 0, "max": 5, "step": 0.01}) 
                    }
                }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "node"

    CATEGORY = chaosaiart_higher.name("image")

    #def colorChange(self,image,Contrast,Color,Brightness,contrast_Override=None,color_Override=None,brightness_Override=None):

    def node(self,image,Contrast,Color,Brightness,Red,Green,Blue):

        ConstrastNum    = Contrast
        ColorNum        = Color
        BrightnessNum   = Brightness 
        RedNum          = Red
        GreenNum        = Green
        BlueNum         = Blue

        imageIN = tensor2pil(image) 

        adjusted_image = chaosaiart_higher.adjust_contrast(imageIN, ConstrastNum)
        adjusted_image = chaosaiart_higher.adjust_saturation(adjusted_image, ColorNum)
        adjusted_image = chaosaiart_higher.adjust_brightness(adjusted_image, BrightnessNum)
        adjusted_image = chaosaiart_higher.adjust_primary_colors(adjusted_image, RedNum, GreenNum, BlueNum)
        
                
        return (pil2tensor(adjusted_image),)
  
#Credit to Pythongosssss 
class chaosaiart_Show_Info:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                #"info": ("STRING", {"forceInput": True}),
                "info": ("STRING", {"multiline": True, "forceInput": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }
  
    INPUT_IS_LIST = True
    RETURN_TYPES = ()
    FUNCTION = "notify"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)

    CATEGORY = chaosaiart_higher.name("main")



    def notify(self, info, unique_id = None, extra_pnginfo=None):
        text = info 
        if unique_id and extra_pnginfo and "workflow" in extra_pnginfo[0]:
            workflow = extra_pnginfo[0]["workflow"]
            node = next((x for x in workflow["nodes"] if str(x["id"]) == unique_id[0]), None)
            if node: 
                node["widgets_values"] = [text]

        return {"ui": {"text": text}, "result": (text,)}
"""
class chaosaiart_video2img2:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "Video_Path": ("STRING", {"default": '', "multiline": False}),   
            }, 
        }
  
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Info",)
    FUNCTION = "node"  

    CATEGORY = chaosaiart_higher.name("image")



    def node(self, Video_Path):  
        FPS_Mode = "Normal"
        
        dir = os.path.dirname(Video_Path)
        video_name = os.path.basename(Video_Path)
        video_name_withoutTyp = os.path.splitext(video_name)[0]
        
        Output_Folder = f'{dir}/{video_name_withoutTyp}'
        info = chaosaiart_higher.video2frame(Video_Path,Output_Folder,FPS_Mode)
        return info,
"""
class chaosaiart_video2img1:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "Video_Path": ("STRING", {"default": '', "multiline": False}),  
                #"Output_Folder": ("STRING", {"default": '', "multiline": False}),
                "FPS_Mode":(["Normal","Low FPS","Lower FPS","Ultra Low FPS"],),
            }, 
        }
  
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Info",)
    FUNCTION = "node"  

    CATEGORY = chaosaiart_higher.name("video")
    
    OUTPUT_NODE = True
 
    #def node(self, Video_Path, FPS_Mode, Output_Folder):
    def node(self, Video_Path, FPS_Mode):
        outPut_Folder = os.path.dirname(self.output_dir)
        outPut_Folder_full = os.path.join(outPut_Folder, "input") 
         
        #outPut_Folder_full = f"{Output_Folder}/input"  
        info = chaosaiart_higher.video2frame(Video_Path,outPut_Folder_full,FPS_Mode)
        return info,



class chaosaiart_Frame_Switch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "activ_frame":("ACTIV_FRAME",), 
                "source_before_frame": (anyType, {}),
                "source_after_frame": (anyType, {}),
                "Switch_frame": ("INT",{"default": 30, "min": 1, "max": 18446744073709551615, "step": 1}),
            }, 
        }
  
    RETURN_TYPES = (anyType,)
    RETURN_NAMES = ("SOURCE",)
    FUNCTION = "node"  

    CATEGORY = chaosaiart_higher.name("switch")

    def node(self, activ_frame,source_before_frame,source_after_frame,Switch_frame): 

        if activ_frame >= Switch_frame:
            return source_after_frame,
        return source_before_frame,
 
 
import threading 

class noob_loading_process:
    def __init__(self, node_name, process_name):
        self.stop_event = threading.Event()
        self.loading_thread = threading.Thread(target=self._animate_loading)
        self.node_Name = node_name
        self.process_name = process_name

    def start(self): 
        self.loading_thread.start()

    def stop(self): 
        self.stop_event.set()
        self.loading_thread.join() 
        print(f"\033[97m{self.node_Name} : Finish                         \n")  # Drucke den Abschlusstext in Weiß

    def _animate_loading(self):
        colors = [
            '\033[94m',  # Hellblau
            '\033[96m',  # Cyan
            '\033[92m',  # Grün
            '\033[32m',  # Hellgrün
        ]
        text_color = '\033[97m'  # Weiß
        color_index = 0
        while not self.stop_event.is_set():
            for frame in ['.  ', '.. ', '...']: 
                print(text_color + self.node_Name + " : " + colors[color_index] + self.process_name + frame, end='\r')
                color_index = (color_index + 1) % len(colors)
                time.sleep(0.5)  # Verzögerung von 0.5 Sekunden 
            print('\033[0m', end='')  # Zurücksetzen auf die Standardfarbe
 

class chaosaiart_img2gif:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "Image_dir": ("STRING", {"default": '', "multiline": False}),  
                "filename_prefix": ("STRING", {"default": 'gif', "multiline": False}),
                "FPS": ("INT",{"default": 10, "min": 1, "max": 18446744073709551615, "step": 1}),
                "Loop":(["Start->END->Start","Start->END"],),
            }, 
            "optional":{
                "merge_folders": ("PATH",),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Info",)
    FUNCTION = "node"  

    OUTPUT_NODE = True

    CATEGORY = chaosaiart_higher.name("video")

    def node(self, Image_dir,filename_prefix,FPS,Loop, merge_folders=None): 

        info = ""
        bilder_ordner = Image_dir if merge_folders is None else merge_folders
        frame_count = FPS

        if not os.path.isdir(bilder_ordner):
            info += f"Folder not exist: {bilder_ordner}" 
            print("chaosaiart_img2gif : "+info)
            return info,
    
        images = []
        empty_Image_dir = True
        for filename in os.listdir(bilder_ordner):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                filepath = os.path.join(bilder_ordner, filename)
                images.append(Image.open(filepath))
                empty_Image_dir = False

        if empty_Image_dir: 
            info += f"No JPG/PNG in: {bilder_ordner}" 
            print("chaosaiart_img2gif : "+info)
            return info, 
        
        folder_preFix = filename_prefix 
        if folder_preFix:
            pre_folder = os.path.join(self.output_dir, folder_preFix) 
            if not os.path.exists(pre_folder):
                os.makedirs(pre_folder)
            output_ordner = pre_folder 
        else:
            output_ordner = self.output_dir 


        file_name = "chaosaiart"
        file_type = "gif"
        
        ausgabedatei = os.path.join(output_ordner, f"{file_name}.{file_type}")
        if os.path.exists(ausgabedatei):
            index = 1 
            while os.path.exists(os.path.join(output_ordner, f'{file_name}_{index}.{file_type}')):
                index += 1
            ausgabedatei = os.path.join(output_ordner, f'{file_name}_{index}.{file_type}') 

        full_image = images if Loop == "Start->END" else images + images[::-1]
         
        print("\n")  
        #TODO: No idea how to implement a progress bar while converting img2gif.
        loading_process = noob_loading_process(chaosaiart_higher.name("main")+" Convert img2gif", "Creating Gif")
        loading_process.start()  
        full_image[0].save(ausgabedatei, save_all=True, append_images=full_image, loop=0, duration=1000//frame_count)
        loading_process.stop()

        #progressbar = tqdm.tqdm(total=anzahl_bilder, desc="Bilder speichern")
        #anzahl_bilder = len(full_image)
        #pbar = tqdm(total=anzahl_bilder, desc="Creating GIF")

        #for index, bild in enumerate(full_image):
            #bild.save(ausgabedatei, save_all=True, append_images=full_image, loop=index, duration=1000//frame_count)
            #progressbar.update()
            #pbar.update(1)

        #pbar.close()
        #with tqdm(total=len(full_image), desc="Creating GIF") as pbar:
            #full_image[0].save(ausgabedatei, save_all=True, append_images=full_image, loop=0, duration=1000//frame_count, progress_callback=lambda i, n: pbar.update(1))
        #for i in range(1, frame_count):
            #full_image[i].save(ausgabedatei, save_all=True, append_images=full_image, loop=i, duration=1000//frame_count)
            #progress_bar.update(1)

        info += f"GIF : {ausgabedatei}"
        return info, 
    
class chaosaiart_img2video:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "Image_dir": ("STRING", {"default": '', "multiline": False}),  
                "filename_prefix": ("STRING", {"default": 'video', "multiline": False}),
                "FPS": ("INT",{"default": 10, "min": 1, "max": 18446744073709551615, "step": 1}),
            }, 
            "optional":{
                "merge_folders": ("PATH",),
            }
        }
  
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Info",)
    FUNCTION = "node"  

    OUTPUT_NODE = True

    CATEGORY = chaosaiart_higher.name("video")



    def node(self, Image_dir,filename_prefix,FPS, merge_folders=None): 
        #TODO: 
        #import shutil
            #import subprocess

        #ffmpeg_path = shutil.which("ffmpeg")
        #if ffmpeg_path is None:
            #raise ProcessLookupError("Could not find ffmpeg")


        file_name = "chaosaiart"
        file_type = "mp4"
        fps = FPS
        bilder_ordner = Image_dir if merge_folders is None else merge_folders

        if not os.path.isdir(bilder_ordner):
            info = "No Folder" 
            print("chaosaiart_img2video : "+info)
            return info,
  
        dateien = os.listdir(bilder_ordner) 
        # Filter PNG & JPG
        bilddateien = [datei for datei in dateien if os.path.isfile(os.path.join(bilder_ordner, datei)) and datei.lower().endswith(('.png', '.jpg', '.jpeg'))]
 
        if not bilddateien:
            info = "No Image"
            print("chaosaiart_img2video : "+info)
            return info,
  
        # Eingabe des Ausgabedateinamens
        folder_preFix = filename_prefix 
        if folder_preFix:
            pre_folder = os.path.join(self.output_dir, folder_preFix) 
            if not os.path.exists(pre_folder):
                os.makedirs(pre_folder)
            output_ordner = pre_folder 
        else:
            output_ordner = self.output_dir 
   
        file_name = "chaosaiart"
        file_type = "mp4"
        
        ausgabedatei = os.path.join(output_ordner, f"{file_name}.{file_type}")
        if os.path.exists(ausgabedatei):
            index = 1 
            while os.path.exists(os.path.join(output_ordner, f'{file_name}_{index}.{file_type}')):
                index += 1
            ausgabedatei = f'{file_name}_{index}.{file_type}'

        output_path = os.path.join(output_ordner, ausgabedatei)
        # Bestimme die Bildgröße anhand des ersten Bildes
        erstes_bild = cv2.imread(os.path.join(bilder_ordner, bilddateien[0]))
        hoehe, breite, _ = erstes_bild.shape

        # Erstelle das Video mit der angegebenen FPS
        #video = cv2.VideoWriter(os.path.join(output_ordner, ausgabedatei), cv2.VideoWriter_fourcc(*'avc1'), fps, (breite, hoehe))
        video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (breite, hoehe))

        info = f'Video Size: {breite}x{hoehe}\nFPS: {fps}' 
        print("chaosaiart_img2video : \n"+info)

        print("Process started, please wait.")
        # Schleife über alle Bilddateien im Ordner
        with tqdm(total=len(bilddateien), desc="Process") as pbar: 
            for datei in bilddateien:
                bildpfad = os.path.join(bilder_ordner, datei)
                bild = cv2.imread(bildpfad)

                video.write(bild)
                pbar.update(1)

        # Schließe das Video und das Fenster
        video.release()
        
        info += f'\nOutput: {output_path}'
        print(f'Output: {output_path}')
        
        return info, 


class chaosaiart_lora: 

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {  
                "lora_name": (folder_paths.get_filename_list("loras"), ),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
            },
            "optional":{
                "add_lora": ("LORA",),
            }
        }
    RETURN_TYPES = ("LORA",)
    FUNCTION = "node"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
    
    CATEGORY = chaosaiart_higher.name("lora")

    def node(self, lora_name, strength_model, strength_clip, add_lora=None): 
           
        loraArray = chaosaiart_higher.add_Lora(add_lora,"positiv",lora_name,strength_model,strength_clip)
        return loraArray,


class chaosaiart_lora_advanced: 

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {  
                "lora_name": (folder_paths.get_filename_list("loras"), ),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                "lora_type": (["Positiv_Prompt","Negativ_Prompt"],)
            },
            "optional":{
                "add_lora": ("LORA",),
                "strength_model_override": ("FLOAT",{"forceInput": True}),
                "strength_clip_override": ("FLOAT",{"forceInput": True}),
            }
        }
    RETURN_TYPES = ("LORA", )
    FUNCTION = "node"

    CATEGORY = chaosaiart_higher.name("lora")

    def node(self, lora_name,lora_type, strength_model, strength_clip, strength_model_override=None,strength_clip_override=None, add_lora=None):
        loraType = "positiv" if lora_type == "Positiv_Prompt" else "negativ"

        strength_model_float =  strength_model if strength_model_override is None else strength_model_override
        strength_clip_float =  strength_clip if strength_clip_override is None else strength_clip_override
         
        loraArray = chaosaiart_higher.add_Lora(add_lora,loraType,lora_name,strength_model_float,strength_clip_float)
        return loraArray,
 
class chaosaiart_zoom_frame: 

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {  
                "Mode": (["This Node will come Later."], ),
            },
            
        }
    RETURN_TYPES = ("ZOOM_FRAME", )
    FUNCTION = "node"

    CATEGORY = chaosaiart_higher.name("image")

    def node(self, Mode):
        return 1,
 
"""
def load_hypernetwork_patch(path, strength):
    sd = comfy.utils.load_torch_file(path, safe_load=True)
    activation_func = sd.get('activation_func', 'linear')
    is_layer_norm = sd.get('is_layer_norm', False)
    use_dropout = sd.get('use_dropout', False)
    activate_output = sd.get('activate_output', False)
    last_layer_dropout = sd.get('last_layer_dropout', False)

    valid_activation = {
        "linear": torch.nn.Identity,
        "relu": torch.nn.ReLU,
        "leakyrelu": torch.nn.LeakyReLU,
        "elu": torch.nn.ELU,
        "swish": torch.nn.Hardswish,
        "tanh": torch.nn.Tanh,
        "sigmoid": torch.nn.Sigmoid,
        "softsign": torch.nn.Softsign,
        "mish": torch.nn.Mish,
    }

    if activation_func not in valid_activation:
        print("Unsupported Hypernetwork format, if you report it I might implement it.", path, " ", activation_func, is_layer_norm, use_dropout, activate_output, last_layer_dropout)
        return None

    out = {}

    for d in sd:
        try:
            dim = int(d)
        except:
            continue

        output = []
        for index in [0, 1]:
            attn_weights = sd[dim][index]
            keys = attn_weights.keys()

            linears = filter(lambda a: a.endswith(".weight"), keys)
            linears = list(map(lambda a: a[:-len(".weight")], linears))
            layers = []

            i = 0
            while i < len(linears):
                lin_name = linears[i]
                last_layer = (i == (len(linears) - 1))
                penultimate_layer = (i == (len(linears) - 2))

                lin_weight = attn_weights['{}.weight'.format(lin_name)]
                lin_bias = attn_weights['{}.bias'.format(lin_name)]
                layer = torch.nn.Linear(lin_weight.shape[1], lin_weight.shape[0])
                layer.load_state_dict({"weight": lin_weight, "bias": lin_bias})
                layers.append(layer)
                if activation_func != "linear":
                    if (not last_layer) or (activate_output):
                        layers.append(valid_activation[activation_func]())
                if is_layer_norm:
                    i += 1
                    ln_name = linears[i]
                    ln_weight = attn_weights['{}.weight'.format(ln_name)]
                    ln_bias = attn_weights['{}.bias'.format(ln_name)]
                    ln = torch.nn.LayerNorm(ln_weight.shape[0])
                    ln.load_state_dict({"weight": ln_weight, "bias": ln_bias})
                    layers.append(ln)
                if use_dropout:
                    if (not last_layer) and (not penultimate_layer or last_layer_dropout):
                        layers.append(torch.nn.Dropout(p=0.3))
                i += 1

            output.append(torch.nn.Sequential(*layers))
        out[dim] = torch.nn.ModuleList(output)

    class hypernetwork_patch:
        def __init__(self, hypernet, strength):
            self.hypernet = hypernet
            self.strength = strength
        def __call__(self, q, k, v, extra_options):
            dim = k.shape[-1]
            if dim in self.hypernet:
                hn = self.hypernet[dim]
                k = k + hn[0](k) * self.strength
                v = v + hn[1](v) * self.strength

            return q, k, v

        def to(self, device):
            for d in self.hypernet.keys():
                self.hypernet[d] = self.hypernet[d].to(device)
            return self

    return hypernetwork_patch(out, strength)
class chaosaiart_HypernetworkLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "hypernetwork_name": (folder_paths.get_filename_list("hypernetworks"), ),
                              "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_hypernetwork"

    CATEGORY = chaosaiart_higher.name("main")

    def load_hypernetwork(self, model, hypernetwork_name, strength):
        hypernetwork_path = folder_paths.get_full_path("hypernetworks", hypernetwork_name)
        model_hypernetwork = model.clone()
        patch = load_hypernetwork_patch(hypernetwork_path, strength)
        if patch is not None:
            model_hypernetwork.set_model_attn1_patch(patch)
            model_hypernetwork.set_model_attn2_patch(patch)
        return (model_hypernetwork,)
"""

# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {

    "chaosaiart_KSampler_a2":                   chaosaiart_KSampler_a2,
    "chaosaiart_KSampler_a1":                   chaosaiart_KSampler_a1,
    "chaosaiart_KSampler_a1a":                  chaosaiart_KSampler_a1a,

    "chaosaiart_video2img1":                    chaosaiart_video2img1,
    "chaosaiart_img2video":                     chaosaiart_img2video,
    "chaosaiart_img2gif":                       chaosaiart_img2gif,

    "chaosaiart_Load_Image_Batch":              chaosaiart_Load_Image_Batch,
    "chaosaiart_Load_Image_Batch_2img":         chaosaiart_Load_Image_Batch_2img,
    "chaosaiart_SaveImage":                     chaosaiart_SaveImage,
    "chaosaiart_EmptyLatentImage":              chaosaiart_EmptyLatentImage, 
    "chaosaiart_adjust_color":                  chaosaiart_adjust_color,

    "chaosaiart_Prompt":                        chaosaiart_Prompt,
    "chaosaiart_Simple_Prompt":                 chaosaiart_Simple_Prompt,
    "chaosaiart_Prompt_Frame":                  chaosaiart_Prompt_Frame,
    "chaosaiart_Prompt_mixer_byFrame":          chaosaiart_Prompt_mixer_byFrame, 
    "chaosaiart_FramePromptCLIPEncode":         chaosaiart_FramePromptCLIPEncode,
    "chaosaiart_MainPromptCLIPEncode":          chaosaiart_MainPromptCLIPEncode,
    "chaosaiart_TextCLIPEncode":                chaosaiart_TextCLIPEncode,
    "chaosaiart_TextCLIPEncode_lora":           chaosaiart_TextCLIPEncode_lora,
    "chaosaiart_convert_Prompt":                chaosaiart_convert_Prompt,

    "chaosaiart_CheckpointPrompt":              chaosaiart_CheckpointPrompt,
    "chaosaiart_CheckpointPrompt2":             chaosaiart_CheckpointPrompt2,
    "chaosaiart_CheckpointLoader":              chaosaiart_CheckpointLoader,
    "chaosaiart_CheckpointPrompt_Frame":        chaosaiart_CheckpointPrompt_Frame,
    "chaosaiart_CheckpointPrompt_FrameMixer":   chaosaiart_CheckpointPrompt_FrameMixer,
    
    "chaosaiart_lora":                          chaosaiart_lora,
    "chaosaiart_lora_advanced":                 chaosaiart_lora_advanced,

    "chaosaiart_oneNode":                       chaosaiart_oneNode,
    "chaosaiart_KSampler1":                     chaosaiart_KSampler1,
    "chaosaiart_KSampler2":                     chaosaiart_KSampler2, 
    "chaosaiart_KSampler3":                     chaosaiart_KSampler3,
    "chaosaiart_Denoising_Switch":              chaosaiart_Denoising_Switch,
   
    "chaosaiart_ControlNetApply":               chaosaiart_ControlNetApply,
    "chaosaiart_ControlNetApply2":              chaosaiart_ControlNetApply2,
    "chaosaiart_ControlNetApply3":              chaosaiart_ControlNetApply3,
    "chaosaiart_controlnet_weidgth":            chaosaiart_controlnet_weidgth,

    "chaosaiart_Number_Counter":                chaosaiart_Number_Counter,
    "chaosaiart_Number":                        chaosaiart_Number,
    "chaosaiart_Number2":                       chaosaiart_Number2,
    "chaosaiart_convert":                       chaosaiart_convert, 

    "chaosaiart_restarter":                     chaosaiart_restarter,
    "chaosaiart_restarter_advanced":            chaosaiart_restarter_advanced,

    "chaosaiart_Any_Switch_small":              chaosaiart_Any_Switch_small,
    "chaosaiart_Any_Switch_Big_Number":         chaosaiart_Any_Switch_Big_Number,
    "chaosaiart_Any_Switch":                    chaosaiart_Any_Switch,
    "chaosaiart_any_array2input_all_small":     chaosaiart_any_array2input_all_small,
    "chaosaiart_any_array2input_all_big":       chaosaiart_any_array2input_all_big,
    "chaosaiart_any_array2input_1Input":        chaosaiart_any_array2input_1Input,
    "chaosaiart_any_input2array_small":         chaosaiart_any_input2array_small,
    "chaosaiart_any_input2array_big":           chaosaiart_any_input2array_big,

    "chaosaiart_reloadIMG_Load":                chaosaiart_reloadIMG_Load, 
    "chaosaiart_reloadIMG_Save":                chaosaiart_reloadIMG_Save,
    "chaosaiart_reloadLatent_Load":             chaosaiart_reloadLatent_Load, 
    "chaosaiart_reloadLatent_Save":             chaosaiart_reloadLatent_Save,

    "chaosaiart_reloadAny_Load":                chaosaiart_reloadAny_Load, 
    "chaosaiart_reloadAny_Save":                chaosaiart_reloadAny_Save,

    "chaosaiart_Number_Switch":                 chaosaiart_Number_Switch,  
    
    "chaosaiart_Show_Info":                     chaosaiart_Show_Info,
   
    "chaosaiart_Frame_Switch":                  chaosaiart_Frame_Switch,
    "chaosaiart_forPreview":                    chaosaiart_forPreview,
    "chaosaiart_zoom_frame":                    chaosaiart_zoom_frame,

    "chaosaiart_merge_Folders":                  chaosaiart_merge_Folders,
 
    "chaosaiart_KSampler_expert_1":             chaosaiart_KSampler_expert_1,
    "chaosaiart_Ksampler_attribut":             chaosaiart_Ksampler_attribut,

   # "chaosaiart_Style_Node":                    chaosaiart_Style_Node,
 
    #"chaosaiart_KSampler_expert_0":             chaosaiart_KSampler_expert_0,
    #"chaosaiart_KSampler4":                     chaosaiart_KSampler4,
    
    #"chaosaiart_KSampler7":                     chaosaiart_KSampler7,
    #"chaosaiart_KSampler5":                     chaosaiart_KSampler5,
    #"chaosaiart_image_loop":                     chaosaiart_image_loop,
  
}
 

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {

    
    "chaosaiart_KSampler_a2":                   "🔶 KSampler txt2video v1", 
    "chaosaiart_KSampler_a1":                   "🔶 KSampler img2video v1",
    "chaosaiart_KSampler_a1a":                  "🔶 KSampler txt2video img2video - Advanced v1",

    "chaosaiart_video2img1":                    "🔶 Convert Video2Img -> Frame",
    "chaosaiart_img2video":                     "🔶 Convert img2video -> mp4",
    "chaosaiart_img2gif":                       "🔶 Convert img2gif -> GIF",

    "chaosaiart_Load_Image_Batch":              "🔶 Load Image Batch",
    "chaosaiart_Load_Image_Batch_2img":         "🔶 Load Image Batch - Advanced",
    "chaosaiart_SaveImage":                     "🔶 AutoSyc Save Image",
    "chaosaiart_adjust_color":                  "🔶 Adjust Color",
    "chaosaiart_EmptyLatentImage":              "🔶 Empty Latent Image - Video Size",


    "chaosaiart_Prompt":                        "🔶 Main Prompt / Prompt Text", 
    "chaosaiart_Simple_Prompt":                 "🔶 Simple Prompt Text",
    "chaosaiart_Prompt_Frame":                  "🔶 Frame Prompt",
    "chaosaiart_Prompt_mixer_byFrame":          "🔶 Prompt mixer by Frame",
    "chaosaiart_TextCLIPEncode":                "🔶 Text Prompt Clip Encode", 
    "chaosaiart_TextCLIPEncode_lora":           "🔶 Text Prompt Clip Endcode +Lora",
    "chaosaiart_FramePromptCLIPEncode":         "🔶 Frame_Prompt Clip Endcode",
    "chaosaiart_MainPromptCLIPEncode":          "🔶 Main_Prompt Clip Endcode", 
    "chaosaiart_convert_Prompt":                "🔶 Convert to Main Prompt",

    "chaosaiart_CheckpointPrompt":              "🔶 Load Checkpoint",
    "chaosaiart_CheckpointPrompt2":             "🔶 Load Checkpoint +Prompt",
    "chaosaiart_CheckpointLoader":              "🔶 Load Checkpoint by Frame",
    "chaosaiart_CheckpointPrompt_Frame":        "🔶 Load Checkpoint +Frame CKPT_PROMPT",
    "chaosaiart_CheckpointPrompt_FrameMixer":   "🔶 CKPT_PROMPT mixer",
    
    "chaosaiart_lora":                          "🔶 Lora +add_lora",
    "chaosaiart_lora_advanced":                 "🔶 Lora Advanced +add_lora",   
    
    "chaosaiart_oneNode":                       "🔶 One Node +Checkpoint +Prompt +Ksampler",
    "chaosaiart_KSampler1":                     "🔶 KSampler txt2img", 
    "chaosaiart_KSampler2":                     "🔶 KSampler img2img",
    "chaosaiart_KSampler3":                     "🔶 KSampler +VAEdecode +Latent",

    "chaosaiart_ControlNetApply":               "🔶 controlnet Apply",
    "chaosaiart_ControlNetApply2":              "🔶 controlnet Apply + Streng Start End",
    "chaosaiart_ControlNetApply3":              "🔶 controlnet Apply Frame",
    "chaosaiart_controlnet_weidgth":            "🔶 Controlnet Weidgth - strenght start end",

    "chaosaiart_Number_Counter":                "🔶 Number Counter",
    "chaosaiart_restarter":                     "🔶 Restart & Activ Frame",
    "chaosaiart_Number":                        "🔶 Number Int",
    "chaosaiart_Number2":                       "🔶 Number Float",
    "chaosaiart_Number_Switch":                 "🔶 One Time Number Switch",  
    
    "chaosaiart_Any_Switch_small":              "🔶 Any Switch", 
    "chaosaiart_Any_Switch_Big_Number":         "🔶 Any Switch (Big)",
    "chaosaiart_Any_Switch":                    "🔶 Any Switch, by Frame", 
    "chaosaiart_reloadIMG_Load":                "🔶 Cache Reloader IMG-> LOAD", 
    "chaosaiart_reloadIMG_Save":                "🔶 Cache Reloader IMG-> SAVE",
    "chaosaiart_reloadLatent_Load":             "🔶 Cache Reloader Latent-> LOAD", 
    
    "chaosaiart_reloadLatent_Save":             "🔶 Cache Reloader Latent-> SAVE",
    "chaosaiart_any_array2input_all_small":     "🔶 Any array2input -> ALL",
    "chaosaiart_any_array2input_all_big":       "🔶 Any array2input -> ALL (Big)",
    "chaosaiart_any_array2input_1Input":        "🔶 Any array2input -> 1 Input",
    "chaosaiart_any_input2array_small":         "🔶 Any input2array",
    "chaosaiart_any_input2array_big":           "🔶 Any input2array (Big)",
    "chaosaiart_reloadAny_Load":                "🔶 Cache Reloader Any-> Load", 
    "chaosaiart_reloadAny_Save":                "🔶 Cache Reloader Any-> Save",
    "chaosaiart_convert":                       "🔶 Convert to Number String Float",
    "chaosaiart_restarter_advanced":            "🔶 Restart & Activ Frame - Advanced",
    "chaosaiart_Show_Info":                     "🔶 Info Display",
    "chaosaiart_Denoising_Switch":              "🔶 Denoise Override (Switch)", 
 
    "chaosaiart_forPreview":                    "🔶 Preview Stacking",
    "chaosaiart_Frame_Switch":                  "🔶 Switch on Frame", 

    "chaosaiart_zoom_frame":                    "🔶 Zoom_Frame - this node will come",
    
    "chaosaiart_merge_Folders":                 "🔶 Merge Folders",

    "chaosaiart_KSampler_expert_1":             "🔶 KSampler Splitted - Expert",
    "chaosaiart_Ksampler_attribut":             "🔶 Main K_ATTRIBUT - Expert", 

    #"chaosaiart_image_loop":                   "🔶 Hold and Repeate one Image",
    #"chaosaiart_Style_Node":                   "🔶 Style Node",
   
   
    #"chaosaiart_KSampler4":                    "🔶 KSampler Advanced", 
    #"chaosaiart_KSampler_expert_0":            "🔶 KSampler expert 0", 
    #"chaosaiart_KSampler7":                    "🔶 KSampler Animation", 
    #"chaosaiart_KSampler5":                    "🔶 KSampler Simple Animation",
   
}
