# ADIP-Final

## Prerequisite 

numpy, opencv-python, PyQt6
```
pip install -r requirement.txt
```

## Try it Now!

With no UI
```
python3 test.py
```

with UI(unstable,testing...)
```
python3 test_ui.py
```

<img width="1847" height="1010" alt="image" src="https://github.com/user-attachments/assets/87874431-7e7f-4158-a0d5-d6b17005a036" />


## TO DO List
1. Add shadow
2. Old photo composition
3. Object edge removal
4. Foreground object refinement(background mis-mask)
5. Add Undo function in UI
6. No defect function on old photo
7. Add execution time for each Level
8. Add AI masking(SAM2) and compare IoU to show on UI

---------------------------------

## Level 1: Foreground Object Extraction and Background Object Removal

### Foreground object precise segmentation:
Accurately extract a specified person from the original image and produce a clean mask with clear boundaries.

### Background object precise removal:
Remove background objects from the image. If parts of the person are missing after removal, appropriate background inpainting should be applied.

### Result

### Foreground object precise segmentation:

Step1: Label Bbox

![ADIP_final_masking](https://github.com/user-attachments/assets/3e75fc37-4e21-4359-b90e-0171d9cfbe41)

<img width="177" height="492" alt="image" src="https://github.com/user-attachments/assets/b1d68900-27f4-4ca8-88a5-985b0a903558" />
<img width="195" height="537" alt="image" src="https://github.com/user-attachments/assets/6af536ac-14b0-4e8b-9d76-1fc4e927328c" />
<img width="170" height="545" alt="image" src="https://github.com/user-attachments/assets/6c296cf0-5a2d-4dfc-975b-22d00fe51b49" />

Step2: Genrate masked image

<img width="550" height="735" alt="image" src="https://github.com/user-attachments/assets/2f7b185c-c98e-4ac0-b182-8cfd979445a9" />
<img width="464" height="590" alt="image" src="https://github.com/user-attachments/assets/bf4ab91f-aadc-44fc-ab98-4e630cd0572e" />


### Background object precise removal:

Still trying the better way to cover the removed area,now is using near pixel to overwrite it
<img width="1281" height="835" alt="image" src="https://github.com/user-attachments/assets/742aba5a-8a41-4364-8614-8fbcb743cf4a" />
<img width="1328" height="864" alt="image" src="https://github.com/user-attachments/assets/991f15e6-67a3-482b-b025-fcedc3ec7528" />

## Level 2: Geometric Correction and Image Composition

This stage focuses on achieving spatial and visual consistency between the foreground and background, including: 

### (a) Perspective and shape correction: 
For foreground images affected by lens distortion, perspective deviation, or scale inconsistency, apply geometric transformations (such as rotation, scaling, mirroring, etc.) so that the foreground appears natural and is correctly aligned within the target background. 

### (b) Image blending (especially at boundaries): 
After compositing the foreground into the target background, carefully process boundaries, transitions, scale, and relative position to ensure the composite appears natural, seamless, and visually consistent.

Case 2:

Before:

![indian_no_smoothing](https://github.com/user-attachments/assets/51d3376d-7138-4655-b994-e49a856427ab)

After:

![indian](https://github.com/user-attachments/assets/c7dfcacf-3707-4fb0-aa02-512d2a4eb43e)

### Result

### (a) Perspective and shape correction: 

![ADIP_final](https://github.com/user-attachments/assets/c0c4ae03-2925-411a-819b-61361bdd129b)

### (b) Image blending (especially at boundaries): 

![saved_result](https://github.com/user-attachments/assets/1d8e1d4d-1c27-4ff1-98e4-ef34175ded4d)


## Level 3: Color Harmonization

Perform color harmonization between the foreground and background images by adjusting color tone, brightness, and overall appearance to achieve a unified and natural visual style in the final composite.

![ADIP_final_shoe_harmony](https://github.com/user-attachments/assets/cac489bd-9132-4dbf-909d-00101e65ed4e)

## (4) Level 4: Manual Light Source Placement

Allow the user to select any position in the image as a light source. Based on this selected position, compute the direction and distance of the light relative to the target object and generate corresponding lighting and shading effects in the image.

![ADIP_final_move_behind](https://github.com/user-attachments/assets/106c634c-4a5c-465e-be87-b227b9fc9f40)






