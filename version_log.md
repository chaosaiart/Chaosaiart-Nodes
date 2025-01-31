# Versionslog:
## Current v1_04_01
You can check your current version by opening the file named 'version_log' in your "comfyUI/custom_node/Chaosaiart-Nodes" folder and looking at what is entered under Current Version<br>

## v1_04

    Added:
        🔶 Prompt - Deepseek <think> Fix 
    
    Details:
        🔶 Prompt - Deepseek <think> Fix 
        It removes the <Think> block of your Deepseek model.
        You need a Deepseek Custom Node, it's just a fix for the prompt.
        
## v1_03

    Feature:
    1. 🔶Lora - Output : Info -> Lora Training Tags + Tag frequency
    2. In all 🔶Prompt Section:
    -  Random Promptpart -> {Random1|Random2|..|RandomXXX} => one will be randomly selected
    
    Added:
    🔶 KSampler txt2video v1
    🔶 KSampler txt2video img2video - Advanced v1
    🔶 Main K_ATTRIBUT - Expert
    🔶 KSampler Splitted - Expert
    🔶 Convert to Main Prompt 
    🔶 Auto None Switch

    Changed: 
    🔶 KSampler txt2video img2video - Advanced v1 -> Input: Splitt_by -> Splitt_by_step


## v1_02

    File:
    WISH_LIST.md

    Added:
    🔶 Convert Img2gif -> GIF
    🔶 KSampler txt2video img2video v1
    🔶 KSampler txt2video img2video - Advanced v1
    🔶 Switch on Frame
    🔶 Preview Stacking

    Changed:
    chaosaiart_video2img -> Output mp4 not h265 mp4 because error. 
    Cache Nodes -> Only one Load Img & Latent with all Funktion.
    Ksampler-img:  
    "Image_Mode":(["Widht = Height","Widescreen / 16:9","Portrait (Smartphone) / 9:16"],),
    "Image_Size":(["360p","480p","HD","Full HD",],),
    "Img2img_input_Size":(["override","resize","crop"],),


## v1_01
    
    Added:
    🔶 KSampler txt2video v1

    Node Change:
    🔶 KSampler txt2video img2video v1 ==> 🔶 KSampler img2video v1
 
    Parameter Changed:
    🔶 Restart & Activ Frame : Add Output Info. -> Frame Activ + Basis Informationen
    🔶 KSampler img2video v1 => ... 
    
    
</details>
