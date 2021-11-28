from PIL import Image
import numpy as np
import cv2
import time
import pathlib

class Character:
    def __init__(self,character):
        self.character = character
        self.asset = self.character['asset']
        self.filename = self.character['filename']
        self.filecount = self.character['filecount']
        self.layout = self.character['layout']
        self.position = self.character['position']
        self.spritesheet = []

        # Generate Spriesheets
        for i in range(self.filecount):
            self.spritesheet.append(Spritesheet(self.filename[i], self.layout[i]))

    def draw(self,surface,pos = None):
        if pos == None:
            pos = [0 for i in self.filecount]
        else:
            assert len(pos) == self.filecount

        for i in range(self.filecount):
            self.spritesheet[i].draw(surface,self.position[i],pos[i])

class Spritesheet:
    def __init__(self,filename,layout = (1,1)):
        #print(str(pathlib.Path(__file__).parent.resolve()) + '/' + filename)
        self.sheet = Image.open(str(pathlib.Path(__file__).parent.resolve()) + '/' + filename)

        self.coords = (0,0)
        self.cols = layout[0]
        self.rows = layout[1]
        self.totalCellCount = self.cols * self.rows

        self.rect = self.sheet.size
        w = self.cellWidth = self.rect[0] // self.cols
        h = self.cellHeight = self.rect[1] // self.rows
        hw, hh = self.cellCenter = (w // 2, h // 2)

        self.cells = list([(i % self.cols * w, int(i / self.cols) * h, i % self.cols * w + w, int(i / self.cols) * h + h) for i in range(self.totalCellCount)])
        self.handle = list([
            (0, 0), (-hw, 0), (-w, 0),
            (0, -hh), (-hw, -hh), (-w, -hh),
            (0, -h), (-hw, -h), (-w, -h)])

    def draw(self,surface,coords = (0,0),cellIndex = 0):
        layer = self.sheet.crop(self.cells[cellIndex])
        surface.paste(layer,coords,mask = layer)

def init_Character():
    head = (11,-1)
    dict = {
        'filecount': 6,
        'asset': [
            'Head',
            'Mouth',
            'Eye',
            'Pupil',
            'Hair',
            'Body'
        ],
        'filename': [
            'Assets/Cypher-Head.png',
            'Assets/LipSyncMouths.png',
            'Assets/Eye-Position.png',
            'Assets/Blinking-Animation.png',
            'Assets/Cypher-Hair.png',
            'Assets/Body-Position.png'
        ],
        'layout': [
            (1, 1),
            (13, 1),
            (12, 3),
            (6,3),
            (1,1),
            (11,1)
        ],
        'position': [
            (head[0]+0,head[1]+0),
            (head[0]+13,head[1]+22),
            (head[0]+8,head[1]+12),
            (head[0]+9,head[1]+12),
            (head[0]+0,head[1]+0),
            (0,0)
        ]
    }
    return dict

def convert_opencv(image):
    numpy_image=np.array(image)  
    opencv_image=cv2.cvtColor(numpy_image, cv2.COLOR_BGRA2RGBA)
    return opencv_image

def generate_frame(character, scale, attributes, images_array):
    surface = Image.new('RGBA', (56, 80)) #creates the output image in Pillow library format
    character.draw(surface, attributes)
    surface = surface.resize((56*scale, 80*scale), resample=Image.BOX) 
    opencv_image = convert_opencv(surface)
    images_array.append(opencv_image)
    return opencv_image

def get_pupil_pos(quadrent):
    quadrent_conversion = [4,3,5,1,7,0,6,2,8]
    return quadrent_conversion[quadrent]

# scale = 10
# c = Character(dict)

# # Generate Surface
# start = time.time()
# surface = Image.new('RGBA', (56, 80))
# c.draw(surface, [0,1,1,1,0,1])
# surface = surface.resize((56*scale, 80*scale), resample=Image.BOX) 

# numpy_image=np.array(surface)  
# opencv_image=cv2.cvtColor(numpy_image, cv2.COLOR_BGRA2RGBA)
# end = time.time()
# print(end - start)

# # Save File
# cv2.imwrite('my.png', opencv_image) 
# #filename = "Frames/Out" + str(0).zfill(6) + ".png"
# #surface.save(filename)