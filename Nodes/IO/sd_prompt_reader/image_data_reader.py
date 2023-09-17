import json
from xml.dom import minidom

import piexif
import pyexiv2
import piexif.helper
from PIL import Image
from PIL.PngImagePlugin import PngInfo

PARAMETER_PLACEHOLDER = "                    "
from .format import (
    A1111,
    EasyDiffusion,
    InvokeAI,
    NovelAI,
    ComfyUI,
    DrawThings,
    SwarmUI,
    Fooocus,
    Primere,
)

class ImageDataReader:
    def __init__(self, file, is_txt: bool = False):
        self._height = None
        self._width = None
        self._info = {}
        self._positive = ""
        self._negative = ""
        self._positive_sdxl = {}
        self._negative_sdxl = {}
        self._setting = ""
        self._raw = ""
        self._tool = ""
        self._parameter_key = ["model", "sampler", "seed", "cfg", "steps", "size"]
        self._parameter = dict.fromkeys(self._parameter_key, PARAMETER_PLACEHOLDER)
        self._is_txt = is_txt
        self._is_sdxl = False
        self._format = ""
        self._parser = None
        self.read_data(file)

    def read_data(self, file):
        def is_json(jsoninput):
            try:
                json.loads(jsoninput)
            except ValueError as e:
                return False
            return True

        if self._is_txt:
            self._raw = file.read()
            self._parser = A1111(raw=self._raw)
            return

        with Image.open(file) as f:
            self._width = f.width
            self._height = f.height
            self._info = f.info
            self._format = f.format
            # swarm format

            '''
            # print("P.debug: " + file + ' - ' + str(self._width) + ' - ' + str(self._height) + ' - ' + str(self._info) + ' - ' + self._format)
            testexif = piexif.load(self._info.get("exif"))
            testresult = piexif.helper.UserComment.load(
                testexif.get("Exif").get(piexif.ExifIFD.UserComment)
            )
            '''

            '''
            if 'Exif.Photo.UserComment' in testresult:
                exif_string = testresult.get('Exif.Photo.UserComment').replace('charset=Unicode', '', 1).strip()
                print('Comment string: ' + exif_string)
            '''

            '''
            {34665: 1118, 270: '{"positive_g": "(beautiful sexy 30 years old (American half-black:1.4) model woman:1.2), (standing in the city:1.3), {half body|full body|portrait|closeup} outdoor color {professional|amateur|mobile} photography, {unhappy|happy|sad|funny|scared|wondering|interested|thinker|serious|cheerful} mood, {open|closed|kissing} mouth, wear {white|brown|red|yellow|green|pink} {leather|luxury|silk|denim} jacket, {blue|white|pink|red|black} {denim|leather|silk} {short|long} pants, (long black straight hair:1.4), {matte|glossy|transparent} fishnet top, large cleavage, natural tits, {Summer|Spring|Winter|Autumn|weekday}, blury background, bokeh, duplicate face, duplicate body, duplicate hair", 
            "negative_g": "porn, sex, nude, nudity, child, childish, Asian-Less-Neg, FastNegativeEmbedding, negative_hand, draw, illustration, telephoto, earrings, necklace", 
            "positive_l": "", 
            "negative_l": "", 
            "positive_refiner": "", 
            "negative_refiner": "", 
            "seed": "596154779", 
            "model_name": "", 
            "sampler_name": "", 
            "original_width": "512", 
            "original_height": "768", 
            "steps": "30", 
            "cfg_scale": 
            "9.0"}'}
            '''

            '''
            p2metadata = pyexiv2.Image(file)
            is_primere = p2metadata.read_exif()
            if 'Exif.Image.ImageDescription' in is_primere:
                primere_exif_string = is_primere.get('Exif.Image.ImageDescription').strip()
                print('Desc. string: ' + primere_exif_string)
                if is_json(primere_exif_string) == True:
                    json_object = json.loads(primere_exif_string)
                    keysList = {'positive_g', 'negative_g', 'positive_l', 'negative_l', 'positive_refiner', 'negative_refiner', 'seed', 'model_name', 'sampler_name'}
                    if not (keysList - json_object.keys()):
                        print('ebben megvan a promere json')            
            '''

            p2metadata = pyexiv2.Image(file)
            is_primere = p2metadata.read_exif()
            if 'Exif.Image.ImageDescription' in is_primere:
                primere_exif_string = is_primere.get('Exif.Image.ImageDescription').strip()
                if is_json(primere_exif_string) == True:
                    json_object = json.loads(primere_exif_string)
                    keysList = {'positive_g', 'negative_g', 'positive_l', 'negative_l', 'positive_refiner', 'negative_refiner', 'seed', 'model_name', 'sampler_name'}
                    if not (keysList - json_object.keys()):
                        self._tool = "Primere"
                        self._parser = Primere(info=json_object)
            else:
                try:
                    exif = json.loads(f.getexif().get(0x0110))
                    if "sui_image_params" in exif:
                        self._tool = "StableSwarmUI"
                        self._parser = SwarmUI(info=exif)
                except TypeError:
                    if f.format == "PNG":
                        # a1111 png format
                        if "parameters" in self._info:
                            self._tool = "A1111 webUI"
                            self._parser = A1111(info=self._info)
                        # easydiff png format
                        elif ("negative_prompt" in self._info or "Negative Prompt" in self._info):
                            self._tool = "Easy Diffusion"
                            self._parser = EasyDiffusion(info=self._info)
                        # invokeai metadata format
                        elif "sd-metadata" in self._info:
                            self._tool = "InvokeAI"
                            self._parser = InvokeAI(info=self._info)
                        # invokeai legacy dream format
                        elif "Dream" in self._info:
                            self._tool = "InvokeAI"
                            self._parser = InvokeAI(info=self._info)
                        # novelai format
                        elif self._info.get("Software") == "NovelAI":
                            self._tool = "NovelAI"
                            self._parser = NovelAI(info=self._info)
                        # comfyui format
                        elif "prompt" in self._info:
                            self._tool = "ComfyUI"
                            self._parser = ComfyUI(
                                info=self._info, width=self._width, height=self._height
                            )
                        # fooocus format
                        elif "Comment" in self._info:
                            try:
                                self._tool = "Fooocus"
                                self._parser = Fooocus(
                                    info=json.loads(self._info.get("Comment"))
                                )
                            except:
                                print("Fooocus format error")
                        # drawthings format
                        elif "XML:com.adobe.xmp" in self._info:
                            try:
                                data = minidom.parseString(
                                    self._info.get("XML:com.adobe.xmp")
                                )
                                data_json = json.loads(
                                    data.getElementsByTagName("exif:UserComment")[0]
                                    .childNodes[1]
                                    .childNodes[1]
                                    .childNodes[0]
                                    .data
                                )
                            except:
                                print("Draw things format error")
                            else:
                                self._tool = "Draw Things"
                                self._parser = DrawThings(info=data_json)
                    elif f.format == "JPEG" or f.format == "WEBP":
                        # fooocus jpeg format
                        if "comment" in self._info:
                            try:
                                self._tool = "Fooocus"
                                self._parser = Fooocus(
                                    info=json.loads(self._info.get("comment"))
                                )
                            except:
                                print("Fooocus format error")
                        else:
                            try:
                                exif = piexif.load(self._info.get("exif")) or {}
                                self._raw = piexif.helper.UserComment.load(
                                    exif.get("Exif").get(piexif.ExifIFD.UserComment)
                                )
                            except TypeError:
                                print("empty jpeg")
                            except Exception:
                                pass
                            else:
                                # easydiff jpeg and webp format
                                if is_json(self._raw) == True:
                                # if self._raw[0] == "{":
                                    self._tool = "Easy Diffusion"
                                    self._parser = EasyDiffusion(raw=self._raw)
                                # a1111 jpeg and webp format
                                else:
                                    self._tool = "A1111 webUI"
                                    self._parser = A1111(raw=self._raw)

    '''
    @staticmethod
    def remove_data(image_file):
        with Image.open(image_file) as f:
            image_data = list(f.getdata())
            image_without_exif = Image.new(f.mode, f.size)
            image_without_exif.putdata(image_data)
            return image_without_exif

    @staticmethod
    def save_image(image_path, new_path, image_format, data=None):
        metadata = None
        if data:
            match image_format:
                case "PNG":
                    metadata = PngInfo()
                    metadata.add_text("parameters", data)
                case "JPEG" | "WEBP":
                    metadata = piexif.dump(
                        {
                            "Exif": {
                                piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(
                                    data, encoding="unicode"
                                )
                            },
                        }
                    )
    
        with Image.open(image_path) as f:
            try:
                match image_format:
                    case "PNG":
                        if data:
                            f.save(new_path, pnginfo=metadata)
                        else:
                            f.save(new_path)
                    case "JPEG":
                        f.save(new_path, quality="keep")
                        if data:
                            piexif.insert(metadata, str(new_path))
                    case "WEBP":
                        f.save(new_path, quality=100, lossless=True)
                        if data:
                            piexif.insert(metadata, str(new_path))
            except:
                print("Save error")
        '''

    def prompt_to_line(self):
        return self._parser.prompt_to_line()

    @property
    def height(self):
        return self._parser.height

    @property
    def width(self):
        return self._parser.width

    @property
    def info(self):
        return self._info

    @property
    def positive(self):
        return self._parser.positive

    @property
    def negative(self):
        return self._parser.negative

    @property
    def positive_sdxl(self):
        return self._parser.positive_sdxl

    @property
    def negative_sdxl(self):
        return self._parser.negative_sdxl

    @property
    def setting(self):
        return self._parser.setting

    @property
    def raw(self):
        return self._parser.raw

    @property
    def tool(self):
        return self._tool

    @property
    def parameter(self):
        return self._parser.parameter

    @property
    def format(self):
        return self._format

    @property
    def is_sdxl(self):
        return self._parser.is_sdxl
