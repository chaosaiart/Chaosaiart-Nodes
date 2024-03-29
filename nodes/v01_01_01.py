  
#TODO: FIXME: NOTIZ: Have to Cach the checkpoint , new loading makes no sence

import cv2 
from tqdm import tqdm

import torch

import os
import sys
import json
import hashlib
import traceback
import math

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

lastCheckpoint_name = ""
cachCheckpoint = []

class anySave:
    def __init__(self, index, saveItem): 
        self.index = index
        self.saveItem = saveItem


class chaosaiart_higher: 
    
    def log(Node,msg,Activ_status):
        if Activ_status:
            print(Node+": "+msg)

    def Debugger(Node, msg):
        if logActiv["Debugger"]:
            print("Debugger: "+Node)
            print(msg)

    def ErrorMSG(Node,msg,Activ_status):
        print("ERROR: "+Node+": "+msg)

    @classmethod    
    def textClipEncode_cacheCheck(cls, clip, text, cache_text, clip_result):
        if cache_text == text:
            if not clip_result == None:
                return clip_result, clip_result,text 
        
        new_result = cls.textClipEncode(clip,text)
        return new_result, new_result, text


    def textClipEncode(clip,text):
        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return [[cond, {"pooled_output": pooled}]]
    
    @classmethod    
    def CKPT_new_or_cache(cls,Checkpoint_name,Cached_CKPT_Name,Cached_CKPT):
        if not Cached_CKPT_Name == Checkpoint_name or Cached_CKPT == None: 
            Cached_CKPT = cls.checkpointLoader(Checkpoint_name)
        return Cached_CKPT, Checkpoint_name

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
    
    def reloader_x(art, index, save, Input):
        
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
            print("Chaosaiart: "+str(art)+"_Cach nr:"+str(index)+" not exist")
            return(None)    

    def check_Text(txt):
        
        try: 
            if txt == None:
                return False
            if txt == "":
                return False
        
            str(txt) 
            return True
        except ZeroDivisionError:
            return False
    
    @classmethod    
    def add_Prompt_txt(cls, txt, txt2):
        if cls.check_Text(txt2):
            if cls.check_Text(txt):
                return txt + "," + txt2
            return txt2
        
        if cls.check_Text(txt):
            return txt
            
        return ""     
    
    @classmethod
    def add_Prompt_txt_byMode(cls, txt, txt2, txt_after=True):
        if txt_after:
            return cls.add_Prompt_txt(txt, txt2)
        return cls.add_Prompt_txt(txt2, txt) 
    
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
    
    """
    @classmethod    
    def lora_by_cache_by_Loop(cls, cache_lora_array,i,lora_name):
        
        lora = None 
        if i < len(cache_lora_array) and cache_lora_array[i] is not None:
            return cls.lora_by_cache(cache_lora_array[i], lora_name)
        return lora
    
    def lora_by_cache(cache_lora, lora_name):
        
        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = None
        if cache_lora is not None:
            if cache_lora[0] == lora_path:
                lora = cache_lora[1] 
        return lora
    """
 
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

        if add_lora == None:
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
            cls.ErrorMSG("Chaosaiart-Load Lora","Can't Use Lora out of Cache",logActiv["lora_load"])
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
            cls.ErrorMSG("Chaosaiart-Load Lora",f"can't use the Lora - {lora_name}",logActiv["lora_load"]) 

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
        if loraArray == None: 
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

                if not cache_loraArray == None: 
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
                    cls.ErrorMSG("Chaosaiart-Load Lora",f"Error, can't Load Lora: {lora_name}",logActiv["lora_load"])
                    cls.log("Chaosaiart-Load Lora","Proof: 1. Lora Exist, 2. Same Base Model as Checkpoint",logActiv["lora_load"])
                    cache_Element = [lora_name, None]
            else: 
                try:
                    e_lora_name = loraArray[i]["lora_name"] if "lora_name" in loraArray[i] else "Unknow" 
                except NameError:  
                    e_lora_name = "Unknow"

                cls.ErrorMSG("Chaosaiart-Load Lora",f"Can't Load one Lora, Name: {e_lora_name}",logActiv["lora_load"])
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

  
        
class chaosaiart_CheckpointPrompt2:
    def __init__(self):
        self.lora_cache = []
        self.Cached_CKPT_Name = ""
        self.Cached_CKPT = None
        self.p_cache_text = ""
        self.n_cache_text = ""
        self.p_cache = None
        self.n_cache = None

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

    CATEGORY = "Chaosaiart/checkpoint"

    def node(self, Checkpoint, Positiv="",Negativ="",add_lora=[],add_positiv_txt = "",add_negativ_txt=""):
        lora = add_lora
  
        ckpt_name = Checkpoint 
        self.Cached_CKPT, self.Cached_CKPT_Name = chaosaiart_higher.CKPT_new_or_cache(ckpt_name,self.Cached_CKPT_Name,self.Cached_CKPT)
        checkpointLoadItem = self.Cached_CKPT   
        
        MODEL   = checkpointLoadItem[0]
        CLIP    = checkpointLoadItem[1]
        VAE     = checkpointLoadItem[2]
        
        sPositiv = chaosaiart_higher.add_Prompt_txt(add_positiv_txt,Positiv)
        sNegativ = chaosaiart_higher.add_Prompt_txt(add_negativ_txt,Negativ)  
         
        
        if not self.lora_cache == []:
             if not chaosaiart_higher.Check_all_loras_in_cacheArray(self.lora_cache,lora): 
                self.lora_cache = [] #Memorey optimization
                
        MODEL, positiv_clip, negativ_clip, self.lora_cache, lora_Info  = chaosaiart_higher.load_lora_by_Array(lora,MODEL,CLIP,self.lora_cache)
         
        PositivOut, self.p_cache , self.p_cache_text = chaosaiart_higher.textClipEncode_cacheCheck(positiv_clip, sPositiv,self.p_cache_text, self.p_cache) 
        NegativOut, self.n_cache, self.n_cache_text = chaosaiart_higher.textClipEncode_cacheCheck(negativ_clip, sNegativ,self.n_cache_text, self.n_cache) 
          
        info  = "checkpoint: "+chaosaiart_higher.path2name(Checkpoint)+"\n" 
        info += f"Positiv:\n{sPositiv}\n"
        info += f"Negativ:\n{sNegativ}\n" 
        info += lora_Info
        
        return (info,MODEL,PositivOut,NegativOut,VAE,) 
 

class chaosaiart_EmptyLatentImage:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                
                #"width": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                #             "height": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                #            "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096})
                "Mode":(["Widescreen / 16:9","Portrait (Smartphone) / 9:16" ],),
                "Size":(["360p","480p","HD","Full HD (Check Info)","QHD (No LATENT, Check Info)","4k (No LATENT, Check Info)","8k (Latent HD, Check Info)",],)
            }
        }
    RETURN_TYPES = ("LATENT","INT","INT","STRING",)
    RETURN_NAMES = ("LATENT","WIDTH","HEIGHT","Info",)
    FUNCTION = "node"


    CATEGORY = "Chaosaiart/image"

    #def generate(self, width, height, batch_size=1):
    def node(self, Mode, Size):
        batch_size=1
        sizeAttribut = {
            "360p":[640,360],
            "480p":[856,480],#4px heigher
            "HD":[1280,720],
            "Full HD (Check Info)":[1920,1080],
            "QHD (No LATENT, Check Info)":[2560,1440],
            "4k (No LATENT, Check Info)":[3840,2160],
            "8k (Latent HD, Check Info)":[7680,4320]
        }
        useSize = sizeAttribut[Size]

        height = useSize[0]
        width  = useSize[1]

        if Mode == "Widescreen / 16:9":   
            height = useSize[1]
            width  = useSize[0]
        
        
        HD_Check = 1080;
        if useSize[1] <= HD_Check:
            info = f"Output:\nlatent=Yes\nwidth:{width}\nheight:{height}"
            latent = torch.zeros([batch_size, 4, height // 8, width // 8], device=self.device)
            if useSize[1] == HD_Check:
                info += "\nBetter Use Upscaler for this Size." 
            print("chaosaiart_EmptyLatentImage: "+info)    
             
            return ({"samples":latent},width,height,info,)
        
        info = f"Output:\nlatent=None\nwidth:{width}\nheight:{height}\nUse Upscalers, checkpoints are not designed for this size + VRAM Problems."
        
        print("chaosaiart_EmptyLatentImage: "+info)
        return (None,width,height,info,)
 

class chaosaiart_CheckpointPrompt:
    def __init__(self):
        self.lora_cache = []
        self.Cached_CKPT_Name = ""
        self.Cached_CKPT = None

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

    CATEGORY = "Chaosaiart/checkpoint"

    def node(self, Checkpoint, positiv_txt="",negativ_txt="",lora=[]):
         
        ckpt_name = Checkpoint
         
        self.Cached_CKPT, self.Cached_CKPT_Name = chaosaiart_higher.CKPT_new_or_cache(ckpt_name,self.Cached_CKPT_Name,self.Cached_CKPT)
        checkpointLoadItem = self.Cached_CKPT  

        MODEL   = checkpointLoadItem[0]
        CLIP    = checkpointLoadItem[1]
        VAE     = checkpointLoadItem[2]   
 
        if not self.lora_cache == []:
             if not chaosaiart_higher.Check_all_loras_in_cacheArray(self.lora_cache,lora): 
                self.lora_cache = [] #Memorey optimization
                
        MODEL, positiv_clip, negativ_clip, self.lora_cache, lora_Info  = chaosaiart_higher.load_lora_by_Array(lora,MODEL,CLIP,self.lora_cache)
         
        PositivOut = chaosaiart_higher.textClipEncode(positiv_clip,positiv_txt)
        NegativOut = chaosaiart_higher.textClipEncode(negativ_clip,negativ_txt)

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

    CATEGORY = "Chaosaiart/prompt"

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

    CATEGORY = "Chaosaiart/prompt"

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

    CATEGORY = "Chaosaiart/checkpoint"

    def node(self, Start_Frame,Checkpoint,Positiv="",Negativ="",lora=[]):  
        return ([Start_Frame,Checkpoint,Positiv,Negativ,lora],)
                
 
class chaosaiart_CheckpointPrompt_FrameMixer:
    def __init__(self):
        self.lora_cache = []
        self.Cached_CKPT_Name = ""
        self.Cached_CKPT = None

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

    CATEGORY = "Chaosaiart/checkpoint"

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
         
        main_positiv = main_prompt[0]
        main_negativ = main_prompt[1]
        main_lora    = main_prompt[2]

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

        #activ_checkpoint_prompt_frame -> [Start_Frame,Checkpoint,Positiv,Negativ]
        activ_checkpoint_prompt_frame = chaosaiart_higher.get_Element_by_Frame(activ_frame,mArray) 

        if activ_checkpoint_prompt_frame == None:
            print("Chaosaiart - CheckpointPrompt_FrameMixer: no checkpoint_prompt_frame with this Activ_Frame. checkpoint_prompt_frame1 will be Used")
            activ_checkpoint_prompt_frame = ckpt_prompt_1

        ckpt_name = activ_checkpoint_prompt_frame[1] 
        self.Cached_CKPT, self.Cached_CKPT_Name = chaosaiart_higher.CKPT_new_or_cache(ckpt_name,self.Cached_CKPT_Name,self.Cached_CKPT)
        checkpointLoadItem = self.Cached_CKPT 


        MODEL   = checkpointLoadItem[0]
        CLIP    = checkpointLoadItem[1]
        VAE     = checkpointLoadItem[2]

        Positiv     = activ_checkpoint_prompt_frame[2]
        Negativ     = activ_checkpoint_prompt_frame[3]
        frame_lora  = activ_checkpoint_prompt_frame[4]
 
        sPositiv = chaosaiart_higher.add_Prompt_txt_byMode(main_positiv,Positiv,True)
        sNegativ = chaosaiart_higher.add_Prompt_txt_byMode(main_negativ,Negativ,True)  

        info  = f"Frame:{activ_frame}\nCheckpoint: {activ_checkpoint_prompt_frame[1]}\nPostiv:\n {sPositiv}\nNegativ:\n {sNegativ}\n" 
        
        #PositivOut = chaosaiart_higher.textClipEncode(CLIP,sPositiv)
        #NegativOut = chaosaiart_higher.textClipEncode(CLIP,sNegativ)
        
        lora = chaosaiart_higher.lora_mainprompt_and_frame(main_lora,frame_lora)
 
        if not self.lora_cache == []:
             if not chaosaiart_higher.Check_all_loras_in_cacheArray(self.lora_cache,lora): 
                self.lora_cache = [] #Memorey optimization
                
        MODEL, positiv_clip, negativ_clip, self.lora_cache, lora_Info  = chaosaiart_higher.load_lora_by_Array(lora,MODEL,CLIP,self.lora_cache)
         

        PositivOut = chaosaiart_higher.textClipEncode(positiv_clip,sPositiv)
        NegativOut = chaosaiart_higher.textClipEncode(negativ_clip,sNegativ)

        info += lora_Info
        #return (info,MODEL,PositivOut,NegativOut,VAE,activ_checkpoint_prompt_frame,) 
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

    CATEGORY = "Chaosaiart/ksampler"

    def node(self, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise, denoise_Override=None):
        denoise = denoise if denoise_Override == None else denoise_Override 
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

    CATEGORY = "Chaosaiart/ksampler"

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

    CATEGORY = "Chaosaiart/ksampler"


 
    @staticmethod
    def vae_encode_crop_pixels(pixels):
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
        return pixels
 
    def node(self, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, image,denoise,denoise_Override=None):
        
        denoise = denoise if denoise_Override == None else denoise_Override 

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
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "vae": ("VAE", ),
                    "denoise": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
                    "empty_Img_width": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                    "empty_Img_height": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                    "empty_Img_batch_size": ("INT", {"default": 1, "min": 1, "max": 4096})
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

    CATEGORY = "Chaosaiart/ksampler"
    
    @staticmethod
    def vae_encode_crop_pixels(pixels):
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
        return pixels
     

    def node(self, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise,empty_Img_width,empty_Img_height,empty_Img_batch_size,latent_Override=None,latent_by_Image_Override=None,denoise_Override=None):
        
        denoise = denoise if denoise_Override == None else denoise_Override 

        if latent_by_Image_Override==None: 
            if latent_Override==None:
                latent = torch.zeros([empty_Img_batch_size, 4, empty_Img_height // 8, empty_Img_width // 8], device=self.device)
                latent_image = {"samples":latent}
                if not denoise==1:
                    print("chaosaiart_KSampler2: set Denoising to 1")
                denoise = 1 
            else: 
                latent_image = latent_Override
        else:
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
                        "empty_Img_width": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                        "empty_Img_height": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                        #"empty_Img_batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                        "steps": ("INT", {"default": 25, "min": 1, "max": 10000}),
                        "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                        "end_at_step": ("INT", {"default": 25, "min": 0, "max": 10000}),
                        "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                        "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                        "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                        "positive": ("CONDITIONING", ),
                        "negative": ("CONDITIONING", ),
                        #"latent_image": ("LATENT", ),
                        #"add_noise": (["enable", "disable"], ),
                        #"return_with_leftover_noise": (["disable", "enable"], ),
                        "vae": ("VAE", ),
                     },
                     "optional":{  
                        "latent_Override": ("LATENT", ), 
                        "latent_by_Image_Override": ("IMAGE", ), 
                    }
                }

    RETURN_TYPES = ("IMAGE","LATENT",) 
    RETURN_NAMES = ("IMAGE","SAMPLES",) 
    FUNCTION = "node"

    CATEGORY = "Chaosaiart/ksampler"

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
             vae,empty_Img_width,empty_Img_height,
             latent_Override = None, latent_by_Image_Override = None,
             denoise=1.0):
        return_with_leftover_noise = "disable"
        empty_Img_batch_size = 1
        add_noise = "enable"

        if latent_by_Image_Override==None: 
            if latent_Override==None:
                latent = torch.zeros([empty_Img_batch_size, 4, empty_Img_height // 8, empty_Img_width // 8], device=self.device)
                latent_image = {"samples":latent}
                start_at_step = 0
            else: 
                latent_image = latent_Override
        else:
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
    
class chaosaiart_KSampler5:
    def __init__(self):
        self.device  = comfy.model_management.intermediate_device()
        self.counter = 1
        #TODO: FIXME: NOTIZ:
        #TODO: FIXME: NOTIZ:
        #TODO: FIXME: NOTIZ: 
        #Something good for more then one Reloader
        #Need a Clean Up function for Cache, maybe implement restarter
        self.reloader_Num = 0 

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                        "restart": ("RESTART",),
                        "model": ("MODEL",), 
                        "Img_width": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                        "Img_height": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}), 
                        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
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

    CATEGORY = "Chaosaiart/ksampler"

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
             vae,empty_Img_width,empty_Img_height,
             restart=0 , Start_Image_Override = None,
             denoise=1.0):
        
        start_at_stepNum = start_at_step 
        return_with_leftover_noise = "disable"
        empty_Img_batch_size = 1
        add_noise = "enable"
        info = "It's frame-by-frame animation.\nPress Queue Prompt for each new frame or use Batch count in extras."


        if self.counter == 1 or restart >= 1:
            self.counter = 1
            if not Start_Image_Override == None:
                #img -> Vae -> Latent  
                pixels = Start_Image_Override
                pixels = self.vae_encode_crop_pixels(pixels)
                t = vae.encode(pixels[:,:,:,:3])
                latent_image = {"samples":t} 
                info += "- Restarted: with Start_Image -\n" 
            else: 
                start_at_stepNum = 0
                latent = torch.zeros([empty_Img_batch_size, 4, empty_Img_height // 8, empty_Img_width // 8], device=self.device)
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

    CATEGORY = "Chaosaiart"

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
    CATEGORY = "Chaosaiart/cache"
 
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
                "vae": ("VAE", ),
                "pre_cache_img_width": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                "pre_cache_img_height": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                "pre_cache_img_batch_size": ("INT", {"default": 1, "min": 1, "max": 4096})
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "node"
    CATEGORY = "Chaosaiart/cache"
 
    def node(self,reloader,vae,pre_cache_img_width,pre_cache_img_height,pre_cache_img_batch_size,restart=0):  

        if restart == 1 or self.is_Started == False:
            self.is_Started = True
            out = vae.decode(torch.zeros([pre_cache_img_batch_size, 4, pre_cache_img_height // 8, pre_cache_img_width // 8], device=self.device)) 
        else:
            out = chaosaiart_higher.reloader_x("img",reloader,False,None)
            if out == None: 
                print("Chaosaiart - reloader: Load Failed")
                out = vae.decode(torch.zeros([pre_cache_img_batch_size, 4, pre_cache_img_height // 8, pre_cache_img_width // 8], device=self.device))
        return ( out, )  

class chaosaiart_reloadIMG_Load2:
    
    def __init__(self): 
        self.is_Started = False
         
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "restart":("RESTART",),
                "image_pre_Save_Out":("IMAGE",),
                "reloader": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "node"
    CATEGORY = "Chaosaiart/cache"
 
    def node(self,reloader,image_pre_Save_Out,restart=0): 

        if restart == 1 or self.is_Started == False:
            self.is_Started = True
            out = image_pre_Save_Out
        else:
            out = chaosaiart_higher.reloader_x("img",reloader,False,None)
            if out == None:
                out = image_pre_Save_Out
        return ( out, ) 


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
    CATEGORY = "Chaosaiart/cache"
 
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
                "pre_cache_width": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                "pre_cache_height": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                "pre_cache_batch_size": ("INT", {"default": 1, "min": 1, "max": 4096})
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("LATENT",)
    FUNCTION = "node"
    CATEGORY = "Chaosaiart/cache"
 
    def node(self,pre_cache_width,pre_cache_height,pre_cache_batch_size,reloader,restart=0): 
 
        if restart == 1 or self.is_Started == False:
            self.is_Started = True
            latent = torch.zeros([pre_cache_batch_size, 4, pre_cache_height // 8, pre_cache_width // 8], device=self.device)
            return {"samples":latent}, 
        else:
            out = chaosaiart_higher.reloader_x("latent",reloader,False,None)
            if out == None:
                print("Chaosaiart - reloader: Load Failed")
                latent = torch.zeros([pre_cache_batch_size, 4, pre_cache_height // 8, pre_cache_width // 8], device=self.device)
                return {"samples":latent},
        return ( out, ) 
    
class chaosaiart_reloadLatent_Load2:
    
    def __init__(self): 
        self.is_Started = False
         
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "restart":("RESTART",), 
                "reloader": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "latent_pre_Save_Out": ("LATENT",),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("LATENT",)
    FUNCTION = "node"
    CATEGORY = "Chaosaiart/cache"
 
    def node(self,latent_pre_Save_Out,reloader,restart=0):  

        if restart == 1 or self.is_Started == False:
            self.is_Started = True
            out = latent_pre_Save_Out
        else:
            out = chaosaiart_higher.reloader_x("latent",reloader,False,None)
            if out == None:
                out = latent_pre_Save_Out
                print("Chaosaiart - reloader: Load Failed")
        return ( out, ) 
    
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
    CATEGORY = "Chaosaiart/cache"
 
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
    CATEGORY = "Chaosaiart/cache"
 
    def node(self,reloader,any_pre_Save_Out,restart=0): 

        if restart == 1 or self.is_Started == False:
            self.is_Started = True
            out = any_pre_Save_Out
        else:
            out = chaosaiart_higher.reloader_x("any",reloader,False,None)
            if out == None:
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

    CATEGORY = "Chaosaiart/prompt"
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

    CATEGORY = "Chaosaiart/prompt"
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

    CATEGORY = "Chaosaiart/prompt"
    def node(self, clip, positiv_txt,negativ_txt,model,lora): 
   
        loraArray = lora
 
        chaosaiart_higher.Debugger("loraA",loraArray) 

        if self.lora_cache == []:
            chaosaiart_higher.log("Chaosaiart-Load Lora","No Lora in Cache.",logActiv["lora_load"])
        else:  
            if not chaosaiart_higher.Check_all_loras_in_cacheArray(self.lora_cache,loraArray): 
                self.lora_cache = [] #Memorey optimization
                chaosaiart_higher.log("Chaosaiart-Load Lora","Clean UP Lora Cache",logActiv["lora_load"])
 
        #TODO: FIXME: NOTIZ: Using Cach to be faster.
        #self.cache_lora_feed, out_Model, positiv_clip , negativ_clip   = chaosaiart_higher.load_lora_by_Array_or_cache(loraArray,model,clip,self.lora_cache)
        #out_Model, positiv_clip, negativ_clip, self.lora_cache, lora_Info
        
        #out_Positiv = chaosaiart_higher.textClipEncode(positiv_clip,positiv_txt)
        #out_Negaitv = chaosaiart_higher.textClipEncode(negativ_clip,negativ_txt)
         


        out_Model, positiv_clip, negativ_clip, self.lora_cache, lora_Info  = chaosaiart_higher.load_lora_by_Array(loraArray,model,clip,self.lora_cache)
        
        out_Positiv = chaosaiart_higher.textClipEncode(positiv_clip,positiv_txt)
        out_Negaitv = chaosaiart_higher.textClipEncode(negativ_clip,negativ_txt)



        #return (chaosaiart_higher.textClipEncode(clip,positiv_txt),chaosaiart_higher.textClipEncode(clip,negativ_txt), )
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

    CATEGORY = "Chaosaiart/prompt"
    def node(self,model, clip, main_prompt): 

        positiv_txt = main_prompt[0]
        negativ_txt = main_prompt[1]
        lora        = main_prompt[2]

        loraArray = lora
 
        chaosaiart_higher.Debugger("loraA",loraArray) 

        if self.lora_cache == []:
            chaosaiart_higher.log("Chaosaiart-Load Lora","No Lora in Cache.",logActiv["lora_load"])
        else:  
            if not chaosaiart_higher.Check_all_loras_in_cacheArray(self.lora_cache,loraArray): 
                self.lora_cache = [] #Memorey optimization
                chaosaiart_higher.log("Chaosaiart-Load Lora","Clean UP Lora Cache",logActiv["lora_load"])
 
        out_Model, positiv_clip, negativ_clip, self.lora_cache, lora_Info  = chaosaiart_higher.load_lora_by_Array(loraArray,model,clip,self.lora_cache)
        
        out_Positiv = chaosaiart_higher.textClipEncode(positiv_clip,positiv_txt)
        out_Negaitv = chaosaiart_higher.textClipEncode(negativ_clip,negativ_txt)
         

        #return (chaosaiart_higher.textClipEncode(clip,positiv_txt),chaosaiart_higher.textClipEncode(clip,negativ_txt), )
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

    CATEGORY = "Chaosaiart/prompt"
    def node(self,model, clip, frame_prompt): 

        positiv_txt = frame_prompt[1][0]
        negativ_txt = frame_prompt[1][1]
        lora        = frame_prompt[1][2]

        loraArray = lora
 
        chaosaiart_higher.Debugger("loraA",loraArray) 

        if self.lora_cache == []:
            chaosaiart_higher.log("Chaosaiart-Load Lora","No Lora in Cache.",logActiv["lora_load"])
        else:  
            if not chaosaiart_higher.Check_all_loras_in_cacheArray(self.lora_cache,loraArray): 
                self.lora_cache = [] #Memorey optimization
                chaosaiart_higher.log("Chaosaiart-Load Lora","Clean UP Lora Cache",logActiv["lora_load"])
 
        out_Model, positiv_clip, negativ_clip, self.lora_cache, lora_Info  = chaosaiart_higher.load_lora_by_Array(loraArray,model,clip,self.lora_cache)
        
        out_Positiv = chaosaiart_higher.textClipEncode(positiv_clip,positiv_txt)
        out_Negaitv = chaosaiart_higher.textClipEncode(negativ_clip,negativ_txt)
         

        #return (chaosaiart_higher.textClipEncode(clip,positiv_txt),chaosaiart_higher.textClipEncode(clip,negativ_txt), )
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

    CATEGORY = "Chaosaiart/restart"
  
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
    
    RETURN_TYPES = ("ACTIV_FRAME", "RESTART",)
    RETURN_NAMES = ("ACTIV_FRAME", "RESTART", )

    FUNCTION = "node"

    CATEGORY = "Chaosaiart/restart"
 

    def node(self, Version):
        
        out_NumCheck = 0
        if not self.Version == Version:
            self.Version = Version
            out_NumCheck = 1
            self.count = 0

        
        self.count += 1

        return (self.count, out_NumCheck,)

 


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
    CATEGORY = "Chaosaiart/switch"
 

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
    CATEGORY = "Chaosaiart/switch"
 

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
    CATEGORY = "Chaosaiart/switch"
 

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
    CATEGORY = "Chaosaiart/switch"
 

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
    CATEGORY = "Chaosaiart/switch"
 

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
    CATEGORY = "Chaosaiart/switch"
 

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
    CATEGORY = "Chaosaiart/logic"
 

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
    CATEGORY = "Chaosaiart/ksampler"
 

    def node(self, First_IMG,Rest_IMG, restart=0):
         
        iNumber = Rest_IMG
        if restart >= 1 or self.started == False:
            self.started = True
            iNumber = First_IMG
 
        info = f"Denoise: {iNumber}"
        return (iNumber,info,)


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
    CATEGORY = "Chaosaiart/switch"
 

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

        source_num = nr_override if not nr_override==None else nr

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
             
        print("test")
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

    
        print("test2")
        info = f"Used Number: {out_NR}" 
        return( aSource[out_NR], info,)
                   
 



            
 
class chaosaiart_Number:
     
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "number_float": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10000000, "step": 0.01}),
                "number_int": ("INT", {"default": 1, "min": 0, "max": 99999999, "step": 1}),
            }
        }

    RETURN_TYPES = ("FLOAT","INT",)
    RETURN_NAMES = ("FLOAT","INT",)
    FUNCTION = "node"
    CATEGORY = "Chaosaiart/logic"

    def node(self,number_float, number_int): 
        return (number_float,number_int,)
     
 
 
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

    CATEGORY = "Chaosaiart/controlnet"

    def node(cls,strength, start, end, strength_override=None, start_override=None, end_override=None):
        iStrength   = strength_override if not strength_override    == None     else strength
        iStart      = start_override    if not start_override       == None     else start
        iEnd        = end_override      if not end_override         == None     else end
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
    CATEGORY = "Chaosaiart/logic"
 
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
            
        if repeat2step == None:
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
            

        info+= f"counter: {counter}"   
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

    CATEGORY = "Chaosaiart/prompt"
 
    def node(self, Prompt="", add_prompt=""): 
        out = chaosaiart_higher.add_Prompt_txt(add_prompt,Prompt)
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

    CATEGORY = "Chaosaiart/prompt"
 
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

    CATEGORY = "Chaosaiart/prompt"
 
    def node(self,start_frame,positiv="",negativ="",add_lora=[],add_positiv="",add_negativ=""):
        
        positivOUT = chaosaiart_higher.add_Prompt_txt(add_positiv,positiv)
        negativOUT = chaosaiart_higher.add_Prompt_txt(add_negativ,negativ)
        return([start_frame,[positivOUT,negativOUT,add_lora]],)
    
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

    CATEGORY = "Chaosaiart/prompt"
    
    def node(self,positiv="",negativ="",add_lora=[],add_positiv="",add_negativ=""):
        positivOUT = chaosaiart_higher.add_Prompt_txt(add_positiv,positiv)
        negativOUT = chaosaiart_higher.add_Prompt_txt(add_negativ,negativ)
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

    CATEGORY = "Chaosaiart/prompt"

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
        
        positiv = chaosaiart_higher.add_Prompt_txt(main_positiv,frame_positiv)
        negativ = chaosaiart_higher.add_Prompt_txt(frame_negativ,main_negativ)
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

    CATEGORY = "Chaosaiart/prompt"

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
   
class chaosaiart_Load_Image_Batch:
    def __init__(self): 
        self.image_history = None 
        self.repeatCount = 0   
        self.counter = 0
        self.Started = False
        self.activ_index = 0
        self.path = ""
 
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["Increasing Index", "by Activ Frame"],),
                "index": ("INT", {"default": 1, "min": 1, "max": 150000, "step": 1}), 
                "path": ("STRING", {"default": '', "multiline": False}), 
                "repeat": ("INT", {"default": 0, "min": 0, "max": 150000, "step": 1}), #Error ohne Sinn 
            },
            "optional": { 
                "restart": ("RESTART",),
                "activ_frame2index": ("ACTIV_FRAME",),
                "repeat_Override": ("REPEAT",),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")


    RETURN_TYPES = ("STRING","IMAGE","REPEAT",)
    RETURN_NAMES = ("Info","IMAGE","REPEAT",)
    FUNCTION = "node"

    CATEGORY = "Chaosaiart/image"

    def node(self, path, restart=0, repeat_Override=None, activ_frame2index=0, repeat=0, index=1, mode="by Activ Frame"):
       
        newImageCheck = False
        
        indexNum   = activ_frame2index if activ_frame2index  >= 1 else index 
        indexNum = indexNum - 1
        repeatNum  = int(repeat_Override) if not repeat_Override==None else repeat 
        restartNum = restart
         
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

                #if mode == "by Activ Frame":   Nothing happen 

                if mode == 'Increasing Index':    
                    indexNum = indexNum + self.counter 
                    self.counter += 1 

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
                    print("Chaosaiart - Load Image Batch: Directory Empty")
                    return None,None,None,
            else:
                self.Started = False
                print("Chaosaiart - Load Image Batch: Directory Path")
                return None,None,None,
                
             
            self.image_history = image 

             
        repeat_OUT = repeatNum - self.repeatCount  
        self.repeatCount += 1 
        info = f"Index: {self.activ_index+1}\nCountdown: {repeat_OUT}" 
        
        if newImageCheck:
            return (info, pil2tensor(image), repeatNum,)
        return (info, pil2tensor(self.image_history), repeatNum,)
  
 
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
                "mode": (["Increasing Index","by Activ Frame"],),
                "index": ("INT", {"default": 1, "min": 1, "max": 150000, "step": 1}), 
                "path": ("STRING", {"default": '', "multiline": False}), 
                "repeat": ("INT", {"default": 0, "min": 0, "max": 150000, "step": 1}), 
            },
            "optional": { 
                "restart": ("RESTART",),
                "activ_frame2index": ("ACTIV_FRAME",),
                "repeat_Override": ("REPEAT",),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")


    RETURN_TYPES = ("STRING", "IMAGE","IMAGE","REPEAT",)
    RETURN_NAMES = ("Info","IMAGE = Index","IMAGE = Index+1","REPEAT",)
    FUNCTION = "node"

    CATEGORY = "Chaosaiart/image"

    def node(self, path,restart=0, repeat_Override=None, activ_frame2index=0, repeat=0, index=0, mode="by Activ Frame"):
      
        newImageCheck = False
        
        indexNum   = activ_frame2index if activ_frame2index  >= 1 else index 
        indexNum = indexNum - 1
        repeatNum  = repeat_Override     if not repeat_Override ==None else repeat 
        restartNum = restart
         
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

                #if mode == "by Activ Frame":   Nothing happen 

                if mode == 'Increasing Index':    
                    indexNum = indexNum + self.counter 
                    indexNum2 = indexNum + 1
                    self.counter += 1
                    #print("Debuging: "+str(self.counter))

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
                    print("Chaosaiart - Load Batch Image: Error: Empty Directory!")
                    return None,None,None,None,
               
            else:
                self.Started = False
                print("Chaosaiart - Load Image Batch: Error: Open Directory Path Failed")
                return None,None,None,None,
                
             
            self.image_history = image 
            self.image_history2 = image2  

             
        repeat_OUT = repeatNum - self.repeatCount 
        self.repeatCount += 1
        info = f"Index: {self.activ_index+1}\nCountdown: {repeat_OUT}" 

        if newImageCheck: 
            return ( info,pil2tensor(image), pil2tensor(image2), repeatNum, )
        return ( info,pil2tensor(self.image_history), pil2tensor(self.image_history2), repeatNum, )
  

  
class chaosaiart_CheckpointLoader:
    def __init__(self):  
        self.lora_cache = []
        self.Cached_CKPT =  None
        self.Cached_CKPT_Name = "" 

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

    CATEGORY = "Chaosaiart/checkpoint"

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
        
         
        ckpt_name = ckpt_1  

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
                        ckpt_name = array[1]
                        break
                    
         
        self.Cached_CKPT, self.Cached_CKPT_Name = chaosaiart_higher.CKPT_new_or_cache(ckpt_name,self.Cached_CKPT_Name,self.Cached_CKPT)
        checkpointLoadItem = self.Cached_CKPT
                
        MODEL   = checkpointLoadItem[0]
        CLIP    = checkpointLoadItem[1]
        VAE     = checkpointLoadItem[2]  

        if not self.lora_cache == []:
             if not chaosaiart_higher.Check_all_loras_in_cacheArray(self.lora_cache,lora): 
                self.lora_cache = [] #Memorey optimization
                
        MODEL, positiv_clip, negativ_clip, self.lora_cache, lora_Info  = chaosaiart_higher.load_lora_by_Array(lora,MODEL,CLIP,self.lora_cache)
         
        PositivOut = chaosaiart_higher.textClipEncode(positiv_clip,positiv_txt)
        NegativOut = chaosaiart_higher.textClipEncode(negativ_clip,negativ_txt)




        #PositivOut = chaosaiart_higher.textClipEncode(CLIP,positiv_txt)
        #NegativOut = chaosaiart_higher.textClipEncode(CLIP,negativ_txt)
        
        info = f"Frame: {activ_frame}\nCheckpoint: {ckpt_name}\nPositiv: {positiv_txt}\nNegativ: {negativ_txt}\n{lora_Info}"
        return (info,MODEL,PositivOut,NegativOut,VAE,) 
  
        #return (info,out[0],out[1],out[2],) 
 




    
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

    CATEGORY = "Chaosaiart/logic"

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

    CATEGORY = "Chaosaiart/controlnet"

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

    CATEGORY = "Chaosaiart/controlnet"

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
                    "strength": ("FLOAT",  {"default": 1, "min": 0, "max": 3, "step": 0.01}),
                    "start": ("FLOAT",  {"default": 0, "min": 0, "max": 1, "step": 0.01}),
                    "end": ("FLOAT",  {"default": 1, "min": 0, "max": 1, "step": 0.01}),
                    "start_Frame":("INT", {"default": 1, "min": 1, "max": 999999999, "step": 1}),
                    "End_Frame":("INT", {"default": 9999, "min": 1, "max": 999999999, "step": 1}),
                }}

    RETURN_TYPES = ("CONDITIONING","CONDITIONING")
    RETURN_NAMES = ("POSITVE", "NEGATIVE")
    FUNCTION = "node"

    CATEGORY = "Chaosaiart/controlnet"

    def node(self, positive, negative, control_net, image, strength, start, end, start_Frame, End_Frame,activ_frame):
        if not ( activ_frame >= start_Frame and  activ_frame < End_Frame ):    
            return (positive, negative)
        
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
                        "Brightness": ("FLOAT", {"default": 1, "min": 0.2, "max": 2, "step": 0.01})
                
                    },
                    #"optional": {
                        #"contrast_Override":("FLOAT",{"forceInput": True}),
                        #"color_Override":("FLOAT",{"forceInput": True}),
                        #"brightness_Override":("FLOAT",{"forceInput": True}),
                
                #}
                }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "node"

    CATEGORY = "Chaosaiart/image"

    #def colorChange(self,image,Contrast,Color,Brightness,contrast_Override=None,color_Override=None,brightness_Override=None):

    def node(self,image,Contrast,Color,Brightness):

        ConstrastNum = Contrast
        ColorNum = Color
        BrightnessNum = Brightness

        imageIN = tensor2pil(image)

        adjusted_image = chaosaiart_higher.adjust_contrast(imageIN, ConstrastNum)
        adjusted_image = chaosaiart_higher.adjust_saturation(adjusted_image, ColorNum)
        adjusted_image = chaosaiart_higher.adjust_brightness(adjusted_image, BrightnessNum)
                
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

    CATEGORY = "Chaosaiart"



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

    CATEGORY = "Chaosaiart/image"



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

    CATEGORY = "Chaosaiart/video"
 
    #def node(self, Video_Path, FPS_Mode, Output_Folder):
    def node(self, Video_Path, FPS_Mode):
        outPut_Folder = os.path.dirname(self.output_dir)
        outPut_Folder_full = os.path.join(outPut_Folder, "input") 
         
        #outPut_Folder_full = f"{Output_Folder}/input"  
        info = chaosaiart_higher.video2frame(Video_Path,outPut_Folder_full,FPS_Mode)
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
                "FPS": ("INT",{"default": 30, "min": 1, "max": 18446744073709551615, "step": 1}),
            }, 
        }
  
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Info",)
    FUNCTION = "node"  

    CATEGORY = "Chaosaiart/video"



    def node(self, Image_dir,filename_prefix,FPS): 
        # Eingabe des Bildordners
        bilder_ordner = Image_dir

        if not os.path.isdir(bilder_ordner):
            info = "No Folder" 
            print("chaosaiart_img2video : "+info)
            return info,

        # Eingabe des Output-Ordners
        output_ordner = os.path.dirname(bilder_ordner)

        # Eingabe des Ausgabedateinamens
        pre_folder = os.path.join(self.output_dir, filename_prefix) 
        ausgabedatei = os.path.join(pre_folder, "chaosaiart") 
        ausgabedatei =  ausgabedatei+".mp4"

        # Eingabe der FPS 
        fps = FPS

        # Liste aller Dateien im Bildordner
        dateien = os.listdir(bilder_ordner)

        # Filtere nur die Bilddateien
        #bilddateien = [datei for datei in dateien if os.path.isfile(os.path.join(bilder_ordner, datei)) and datei.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        bilddateien = [datei for datei in dateien if os.path.isfile(os.path.join(bilder_ordner, datei)) and datei.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Prüfe, ob im Bildordner Bilddateien vorhanden sind
        if not bilddateien:
            info = "No Image"
            print("chaosaiart_img2video : "+info)
            return info,

        # Prüfe, ob der Ausgabeordner existiert, andernfalls erstelle ihn
        if not os.path.exists(output_ordner):
            os.makedirs(output_ordner)

        # Prüfe, ob die Ausgabedatei bereits existiert
        ausgabepfad = os.path.join(output_ordner, ausgabedatei)
        if os.path.exists(ausgabepfad):
            index = 1
            dateiname, dateiendung = os.path.splitext(ausgabedatei)
            while os.path.exists(os.path.join(output_ordner, f'{dateiname}_{index}{dateiendung}')):
                index += 1
            ausgabedatei = f'{dateiname}_{index}{dateiendung}'

        # Bestimme die Bildgröße anhand des ersten Bildes
        erstes_bild = cv2.imread(os.path.join(bilder_ordner, bilddateien[0]))
        hoehe, breite, _ = erstes_bild.shape

        # Erstelle das Video mit der angegebenen FPS
        video = cv2.VideoWriter(os.path.join(output_ordner, ausgabedatei), cv2.VideoWriter_fourcc(*'mp4v'), fps, (breite, hoehe))


        info = f'Video Size: {breite}x{hoehe}\nFPS: {fps}' 
        print("chaosaiart_img2video : \n"+info)

        print("Process started, please wait.")
        # Schleife über alle Bilddateien im Ordner
        with tqdm(total=len(bilddateien), desc="Process") as pbar:
        #with tqdm(total=(bilddateien), desc="Process") as pbar:
            for datei in bilddateien:
                bildpfad = os.path.join(bilder_ordner, datei)
                bild = cv2.imread(bildpfad)

                # Füge das aktuelle Bild zum Video hinzu
                video.write(bild)

                # Zeige das aktuelle Bild im Fenster an (optional)
                #cv2.imshow('Bild zum Video', bild)
                #cv2.waitKey(1)  
                # Warte 1 Millisekunde zwischen den Bildern# Aktualisiere den Fortschrittsbalken
                pbar.update(1)

        # Schließe das Video und das Fenster
        video.release()
        
        info += f'\nOutput: {ausgabedatei}'
        print(f'Output: {ausgabedatei}')
        
        return info,
        #cv2.destroyAllWindows()


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

    CATEGORY = "Chaosaiart/lora"

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

    CATEGORY = "Chaosaiart/lora"

    def node(self, lora_name,lora_type, strength_model, strength_clip, strength_model_override=None,strength_clip_override=None, add_lora=None):
        loraType = "positiv" if lora_type == "Positiv_Prompt" else "negativ"

        strength_model_float =  strength_model if strength_model_override == None else strength_model_override
        strength_clip_float =  strength_clip if strength_clip_override == None else strength_clip_override
         
        loraArray = chaosaiart_higher.add_Lora(add_lora,loraType,lora_name,strength_model_float,strength_clip_float)
        return loraArray,
 
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

    CATEGORY = "loaders"

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
    "chaosaiart_video2img1":                    chaosaiart_video2img1,
    "chaosaiart_img2video":                     chaosaiart_img2video,

    "chaosaiart_adjust_color":                  chaosaiart_adjust_color,
    "chaosaiart_Load_Image_Batch":              chaosaiart_Load_Image_Batch,
    "chaosaiart_Load_Image_Batch_2img":         chaosaiart_Load_Image_Batch_2img,
    "chaosaiart_SaveImage":                     chaosaiart_SaveImage,
    "chaosaiart_EmptyLatentImage":              chaosaiart_EmptyLatentImage, 
    
    "chaosaiart_Prompt":                        chaosaiart_Prompt,
    "chaosaiart_Simple_Prompt":                 chaosaiart_Simple_Prompt,
    "chaosaiart_Prompt_Frame":                  chaosaiart_Prompt_Frame,
    "chaosaiart_Prompt_mixer_byFrame":          chaosaiart_Prompt_mixer_byFrame,
    "chaosaiart_FramePromptCLIPEncode":         chaosaiart_FramePromptCLIPEncode,
    "chaosaiart_MainPromptCLIPEncode":          chaosaiart_MainPromptCLIPEncode,
    "chaosaiart_TextCLIPEncode":                chaosaiart_TextCLIPEncode,
    "chaosaiart_TextCLIPEncode_lora":           chaosaiart_TextCLIPEncode_lora,

    "chaosaiart_CheckpointPrompt":              chaosaiart_CheckpointPrompt,
    "chaosaiart_CheckpointPrompt2":             chaosaiart_CheckpointPrompt2,
    "chaosaiart_CheckpointLoader":              chaosaiart_CheckpointLoader,
    "chaosaiart_CheckpointPrompt_Frame":        chaosaiart_CheckpointPrompt_Frame,
    "chaosaiart_CheckpointPrompt_FrameMixer":   chaosaiart_CheckpointPrompt_FrameMixer,
    
    "chaosaiart_lora":                          chaosaiart_lora,
    "chaosaiart_lora_advanced":                 chaosaiart_lora_advanced,

    "chaosaiart_KSampler1":                     chaosaiart_KSampler1,
    "chaosaiart_KSampler2":                     chaosaiart_KSampler2, 
    "chaosaiart_KSampler3":                     chaosaiart_KSampler3,
    "chaosaiart_KSampler4":                     chaosaiart_KSampler4,
    #"chaosaiart_KSampler5":                     chaosaiart_KSampler5,
    "chaosaiart_Denoising_Switch":              chaosaiart_Denoising_Switch,
   
    "chaosaiart_ControlNetApply":               chaosaiart_ControlNetApply,
    "chaosaiart_ControlNetApply2":              chaosaiart_ControlNetApply2,
    "chaosaiart_ControlNetApply3":              chaosaiart_ControlNetApply3,
    "chaosaiart_controlnet_weidgth":            chaosaiart_controlnet_weidgth,

    "chaosaiart_Number_Counter":                chaosaiart_Number_Counter,
    "chaosaiart_Number":                        chaosaiart_Number,
    "chaosaiart_convert":                       chaosaiart_convert, 

    "chaosaiart_restarter":                     chaosaiart_restarter,
    "chaosaiart_restarter_advanced":            chaosaiart_restarter_advanced,

    "chaosaiart_Any_Switch":                    chaosaiart_Any_Switch,
    "chaosaiart_Any_Switch_Big_Number":         chaosaiart_Any_Switch_Big_Number,
    "chaosaiart_any_array2input_all_small":     chaosaiart_any_array2input_all_small,
    "chaosaiart_any_array2input_all_big":       chaosaiart_any_array2input_all_big,
    "chaosaiart_any_array2input_1Input":        chaosaiart_any_array2input_1Input,
    "chaosaiart_any_input2array_small":         chaosaiart_any_input2array_small,
    "chaosaiart_any_input2array_big":           chaosaiart_any_input2array_big,

    "chaosaiart_reloadIMG_Load":                chaosaiart_reloadIMG_Load,
    "chaosaiart_reloadIMG_Load2":               chaosaiart_reloadIMG_Load2,
    "chaosaiart_reloadIMG_Save":                chaosaiart_reloadIMG_Save,
    "chaosaiart_reloadLatent_Load":             chaosaiart_reloadLatent_Load,
    "chaosaiart_reloadLatent_Load2":            chaosaiart_reloadLatent_Load2,
    "chaosaiart_reloadLatent_Save":             chaosaiart_reloadLatent_Save,

    "chaosaiart_reloadAny_Load":                chaosaiart_reloadAny_Load, 
    "chaosaiart_reloadAny_Save":                chaosaiart_reloadAny_Save,

    "chaosaiart_Number_Switch":                 chaosaiart_Number_Switch,  
    
    "chaosaiart_Show_Info":                     chaosaiart_Show_Info,
   # "chaosaiart_Style_Node":                    chaosaiart_Style_Node,
  
}
 

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "chaosaiart_video2img1":                    "🔶 Video2Img -> Frame",
    "chaosaiart_img2video":                     "🔶 img2video -> Video",

    "chaosaiart_adjust_color":                  "🔶 Adjust Color - Contrast, Bright, Color",
    "chaosaiart_Load_Image_Batch":              "🔶 Load Image Batch",
    "chaosaiart_Load_Image_Batch_2img":         "🔶 Load Image Batch - Advanced",
    "chaosaiart_SaveImage":                     "🔶 AutoSyc Save Image",

    "chaosaiart_Prompt":                        "🔶 Prompt Text / Main Prompt", 
    "chaosaiart_Simple_Prompt":                 "🔶 Simple Prompt Text",
    "chaosaiart_Prompt_Frame":                  "🔶 Frame Prompt",
    "chaosaiart_Prompt_mixer_byFrame":          "🔶 Prompt mixer by Frame",
    "chaosaiart_TextCLIPEncode":                "🔶 Text Prompt Clip Encode", 
    "chaosaiart_TextCLIPEncode_lora":           "🔶 Text Prompt Clip Endcode +Lora",
    "chaosaiart_FramePromptCLIPEncode":         "🔶 Frame_Prompt Clip Endcode",
    "chaosaiart_MainPromptCLIPEncode":          "🔶 Main_Prompt Clip Endcode",

    "chaosaiart_CheckpointPrompt":              "🔶 Load Checkpoint",
    "chaosaiart_CheckpointPrompt2":             "🔶 Load Checkpoint +Prompt",
    "chaosaiart_CheckpointLoader":              "🔶 Load Checkpoint by Frame",
    "chaosaiart_CheckpointPrompt_Frame":        "🔶 Load Checkpoint +Frame CKPT_PROMPT",
    "chaosaiart_CheckpointPrompt_FrameMixer":   "🔶 Checkpoint mixer by Frame",
    
    "chaosaiart_lora":                          "🔶 Lora +add_lora",
    "chaosaiart_lora_advanced":                 "🔶 Lora Advanced +add_lora",   
    
    "chaosaiart_KSampler1":                     "🔶 KSampler txt2img", 
    "chaosaiart_KSampler2":                     "🔶 KSampler img2img",
    "chaosaiart_KSampler3":                     "🔶 KSampler +VAEdecode +Latent",
    "chaosaiart_KSampler4":                     "🔶 KSampler Advanced",
    #"chaosaiart_KSampler5":                     "🔶 KSampler Simple Animation",

    "chaosaiart_ControlNetApply":               "🔶 controlnet Apply",
    "chaosaiart_ControlNetApply2":              "🔶 controlnet Apply + Streng Start End",
    "chaosaiart_ControlNetApply3":              "🔶 controlnet Apply Frame",
    "chaosaiart_controlnet_weidgth":            "🔶 Controlnet Weidgth - strenght start end",

    "chaosaiart_Number_Counter":                "🔶 Number Counter",
    "chaosaiart_restarter":                     "🔶 Restart & Activ Frame",
    "chaosaiart_Number":                        "🔶 Number Int float",
    "chaosaiart_Number_Switch":                 "🔶 One Time Number Switch",  
    
    "chaosaiart_Any_Switch":                    "🔶 Any Switch", 
    "chaosaiart_Any_Switch_Big_Number":         "🔶 Any Switch (Big)",
    "chaosaiart_reloadIMG_Load":                "🔶 Cache Reloader IMG-> LOAD",
    "chaosaiart_reloadIMG_Load2":               "🔶 Cache Reloader IMG-> LOAD +Img",
    "chaosaiart_reloadIMG_Save":                "🔶 Cache Reloader IMG-> SAVE",
    "chaosaiart_reloadLatent_Load":             "🔶 Cache Reloader Latent-> LOAD",
    "chaosaiart_reloadLatent_Load2":            "🔶 Cache Reloader Latent-> LOAD +Latent",
    
    "chaosaiart_reloadLatent_Save":             "🔶 Cache Reloader Latent-> SAVE",
    "chaosaiart_any_array2input_all_small":     "🔶 Any array2input -> ALL",
    "chaosaiart_any_array2input_all_big":       "🔶 Any array2input -> ALL (Big)",
    "chaosaiart_any_array2input_1Input":        "🔶 Any array2input -> 1 Input",
    "chaosaiart_any_input2array_small":         "🔶 Any input2array",
    "chaosaiart_any_input2array_big":           "🔶 Any input2array (Big)",
    "chaosaiart_reloadAny_Load":                "🔶 Cache Reloader Any-> Load", 
    "chaosaiart_reloadAny_Save":                "🔶 Cache Reloader Any-> Save",
    "chaosaiart_convert":                       "🔶 Convert to Number String Float",
    "chaosaiart_restarter_advanced":            "🔶 Restart & Activ - Advanced",
    "chaosaiart_Show_Info":                     "🔶 Info Display",
    "chaosaiart_Denoising_Switch":              "🔶 Denoise Override (Switch)", 
    "chaosaiart_EmptyLatentImage":              "🔶 Empty Latent Image - Video Size",
   # "chaosaiart_Style_Node":                    "🔶 Style Node",
   
}
