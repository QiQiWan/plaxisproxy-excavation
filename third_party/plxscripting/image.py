"""
Purpose: Provide objects that represent an image created by the server.

Subversion data:
    $Id: image.py 19791 2015-07-16 13:26:27Z tj $
    $URL: https://tools.plaxis.com/svn/sharelib/trunk/PlxObjectLayer/Server/plxscripting/image.py $

Copyright (c) Plaxis bv. All rights reserved.

Unless explicitly acquired and licensed from Licensor under another
license, the contents of this file are subject to the Plaxis Public
License ("PPL") Version 1.0, or subsequent versions as allowed by the PPL,
and You may not copy or use this file in either source code or executable
form, except in compliance with the terms and conditions of the PPL.

All software distributed under the PPL is provided strictly on an "AS
IS" basis, WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESS OR IMPLIED, AND
LICENSOR HEREBY DISCLAIMS ALL SUCH WARRANTIES, INCLUDING WITHOUT
LIMITATION, ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE, QUIET ENJOYMENT, OR NON-INFRINGEMENT. See the PPL for specific
language governing rights and limitations under the PPL.
"""

import io
import codecs


TYPE_NAME_IMAGE = 'image/png'


class ImageBytesWrapper(object):
    def __init__(self, image_bytes):
        super(ImageBytesWrapper, self).__init__()
        self._image_bytes = image_bytes

    @property
    def image(self):
        raise RuntimeError("Can't return Image object. Pillow not installed.")

    @property
    def bytes(self):
        return self._image_bytes

    def save(self, path):
        with open(path, 'wb') as image_file:
            image_file.write(self._image_bytes)


class PILImageWrapper(object):
    def __init__(self, image):
        super(PILImageWrapper, self).__init__()
        self._image = image

    @property
    def image(self):
        return self._image

    @property
    def bytes(self):
        return self._image.tobytes()

    def save(self, path):
        self._image.save(path)


def create_image(json_object):
    if 'data' not in json_object:
        raise Exception("JSON for image must contain 'data' property.")

    image_bytes_base64 = json_object['data'].encode('ascii')
    image_bytes = codecs.decode(image_bytes_base64, 'base64')

    try:
        from PIL import Image
        image = Image.open(io.BytesIO(image_bytes))
        return PILImageWrapper(image)
    except ImportError:
        return ImageBytesWrapper(image_bytes)
