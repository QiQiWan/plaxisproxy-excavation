# -*- coding: utf-8 -*-
"""
  Purpose:
    Handles the transparent encryption and decryption of HTTP
    communications according to Plaxis rules.

    An encrypted request must consist of:
        - a Base64-encoded code, which is the initialization vector used in
          the decryption
        - a Blowfish-encrypted and then Base64-encoded JSON object
        - the JSON object must include a reply code. After handling the
          request, the response sent to the caller must include the reply
          code. This way the caller can verify that the handler successfully
          decrypted the message (i.e. the handler was not spoofed).

    An unencrypted request consists of:
        - a JSON object

    An encrypted response consists of:
        - a JSON object with two fields:
            - response: Blowfish-encrypted and then Base64-encoded JSON object,
                        including the reply code.
            - code: Base64-encoded initialization vector needed for decrypting
                    the response

    An unecrypted response consists of:
        - a JSON object


  Subversion data:
    $Id: encryption.py 11612 2013-04-02 14:17:47Z ac $
    $URL: https://tools.plaxis.com/svn/sharelib/trunk/PlxObjectLayer/Server/RemoteLicenceServer/rsauth/encryption.py $

  Copyright (c) Plaxis BV. All rights reserved.

"""

from Crypto.Cipher import Blowfish
# Adjusting Blowfish key size minimum value so it can be compatible with Delphi implementation that allows smaller keys
Blowfish.key_size = range(1, 56+1)
from Crypto import Random
import base64
import json
import uuid
import string

# request parameters
REQUEST_DATA = 'RequestData'
CODE = 'Code'

# JSON fields
REPLY_CODE = 'ReplyCode'
RESPONSE = 'Response'

blocksize = Blowfish.block_size

def make_cipher(key, initialization_vector):
    """"The PyCrypto docs state about encryption modes:
        The simplest is Electronic Code Book (or ECB) mode.
        In this mode, each block of plaintext is simply
        encrypted to produce the ciphertext. This mode can
        be dangerous, because many files will contain
        patterns greater than the block size; for example,
        the comments in a C program may contain long strings
        of asterisks intended to form a box. All these
        identical blocks will encrypt to identical ciphertext;
        an adversary may be able to use this structure to
        obtain some information about the text.
        ...
        One mode is Cipher Block Chaining (CBC mode); another
        is Cipher FeedBack (CFB mode). CBC mode still encrypts
        in blocks, and thus is only slightly slower than ECB
        mode. CFB mode encrypts on a byte-by-byte basis, and
        is much slower than either of the other two modes.

    For this reason we will use CBC.

    """
    if len(initialization_vector) != blocksize: # prevent crashing in case of malicious init vector
        initialization_vector = bytearray(blocksize)
    return Blowfish.new(key.encode('utf-8'), Blowfish.MODE_CBC, initialization_vector)


def encrypt(datastring, key):
    """Encrypts the data string using the specified key and
    returns a tuple containing an initialization vector
    and the encrypted data respectively.
    The initialization vector (iv) is needed for decrypting
    the data again.
    All resulting data is also encoded with b64 for ease of
    use over HTTP.

    """
    iv = Random.new().read(blocksize)
    cipher = make_cipher(key, iv)

    # Needed because of the use of rstrip() in decrypt
    if datastring.endswith(tuple(string.whitespace)):
        raise Exception("String ending in whitespace can't be encrypted correctly.")

    data = datastring.encode('utf-8') # ensure that the encrypted data comes after decryption in a predictable encoding (utf-8)
    padding_length = blocksize - len(data) % blocksize
    padding = b' ' * padding_length
    encrypted_data = cipher.encrypt(data + padding)
    return (base64.b64encode(encrypted_data).decode('ascii'), base64.b64encode(iv).decode('ascii'))


def base64decode_safe(encoded_string):
    """The base64-encoded data comes from potentially
    untrusted sources. Therefore don't assume it's
    correct.

    """
    try:
        return base64.b64decode(encoded_string)
    except:
        return ''


def decrypt(encrypted_data, initialization_vector, key):
    """The data and init vector must be base64 encoded!"""
    cipher = make_cipher(key, base64decode_safe(initialization_vector))
    try: # encrypted data comes from untrusted source and may be corrupted
        res = cipher.decrypt(base64decode_safe(encrypted_data))
        res = res.decode('utf-8')
        return res.rstrip()
    except:
        return ''


ERR_INVALID_CONTENT_FOR = 'Invalid content for %s'
ERR_MISSING_ITEM = 'Missing: %s'
ERR_BAD_REPLY_CODE = 'Bad reply code'
ERR_INVALID_RESPONSE_DATA_JSON = 'Invalid response data JSON'
ERR_NO_EXPECTED_REPLY_CODE_DEFINED = 'No expected reply code defined'


class EncryptionHandler():
    """ An object capable of handling requests, with transparent
    encryption/decryption facilities.

    It can be used as:
        - decryptor, in which case it will decrypt the specified
          request data
        - encryptor, in which case it will encrypt the specified
          response data

    The handler takes care of the reply_code on its own, i.e. if first
    called as decryptor and later as encryptor, it will take the
    reply_code it found while decrypting and automatically add it to
    the encrypted response data.

    Children will assume that if the key is unspecified, encryption
    is disabled.

    """
    def __init__(self, key):
        """If the key is not specified, no encryption/decryption will
        be performed.

        """
        self.key = key # used for encrypting/decrypting
        self.data = {}
        self.reply_code = ''


class EncryptionHandlerServer(EncryptionHandler):
    """Offers facilities to a server: can decrypt an incoming request and
    encrypt an outgoing response.

    """
    def interpret_request(self, code, request_data):
        """Interprets the request data using the current key and the
        specified code (initialization vector).

        Returns a tuple with in the first position a boolean indicating
        success (True) or failure (False) and in the second position an
        error message (in case of failure only).

        Sets internally the reply_code for later use (if applicable).

        """
        if self.key: # otherwise treat as unencrypted data
            decrypted_data = decrypt(request_data, code, self.key)
        else:
            decrypted_data = request_data

        self.data = {}
        try:
            self.data = json.loads(decrypted_data)
        except:
            return (False, ERR_INVALID_CONTENT_FOR % (REQUEST_DATA))


        # encrypted communications are supposed to have a reply code so the
        # caller can verify that the callee actually decrypted the message
        # (the callee must include the reply code in its encrypted response)
        if self.key:
            try:
                self.reply_code = self.data[REPLY_CODE]
                del self.data[REPLY_CODE] # is infrastructure stuff, don't expose to the caller
            except:
                self.data = {}
                return (False, ERR_MISSING_ITEM % (REPLY_CODE))

        return (True, '')

    def build_response(self, response_dict):
        if self.key:
            if not REPLY_CODE in response_dict:
                response_dict[REPLY_CODE] = self.reply_code

            enc_data, iv = encrypt(json.dumps(response_dict), self.key)
            container_response_dict = {CODE: iv, RESPONSE: enc_data}
        else:
            container_response_dict = response_dict
        response_data = json.dumps(container_response_dict)
        return response_data


class EncryptionHandlerClient(EncryptionHandler):
    """Offers facilities to a client: can encrypt an outgoing request and
    decrypt the incoming response.

    """
    def build_request_dict(self, request_parameters):
        """Builds an (encrypted) request dictionary, including a
        reply code. The reply code is remembered for checking
        purposes in the interpreting of the repsonse.

        """
        if self.key:
            if REPLY_CODE not in request_parameters:
                self.reply_code = '{%s}' % (str(uuid.uuid4()).upper())
                request_parameters[REPLY_CODE] = self.reply_code
            else:
                self.reply_code = request_parameters[REPLY_CODE]
            encrypted_data, code = encrypt(json.dumps(request_parameters), self.key)
            return {REQUEST_DATA: encrypted_data, CODE: code}
        else:
            self.reply_code = ''
            return {REQUEST_DATA: json.dumps(request_parameters)}


    def interpret_response(self, response_data):
        """Interprets the response data using the current key.

        Checks the reply_code.

        Returns a tuple with in the first position a boolean indicating
        success (True) or failure (False) and in the second position an
        error message (in case of failure only).

        """
        self.data = {}
        try:
            container_response_dict = json.loads(response_data)
        except:
            return (False, ERR_INVALID_RESPONSE_DATA_JSON)

        if self.key:
            if not self.reply_code:
                return (False, ERR_NO_EXPECTED_REPLY_CODE_DEFINED)

            # untangle the response
            try:
                code = container_response_dict[CODE]
            except:
                return (False, ERR_MISSING_ITEM % (CODE))

            try:
                enc_response = container_response_dict[RESPONSE]
            except:
                return (False, ERR_MISSING_ITEM % (RESPONSE))

            response = decrypt(enc_response, code, self.key)
            try:
                response_dict = json.loads(response)
            except:
                return (False, ERR_INVALID_CONTENT_FOR % (RESPONSE))

            if REPLY_CODE not in response_dict:
                return (False, ERR_MISSING_ITEM % (REPLY_CODE))
            if response_dict[REPLY_CODE] != self.reply_code:
                return (False, ERR_BAD_REPLY_CODE)
            del response_dict[REPLY_CODE] # is infrastructure stuff, don't expose to the caller
        else: # not encrypted
            response_dict = container_response_dict

        self.data = response_dict
        return (True, '')





##s = u'概要hello world'
##encs, iv = encrypt(s)
##print(s)
##print(iv)
##print(encs)
##print(decrypt(encs, iv))
##print(decrypt('FPx/CpcvSEkJJ4Nfa6yHqP0V975wBMmHrMQSMKbWJP+8b9TM/CUcHuGGCYKKkv' + \
##      '0zSwxTmkDuYh0lmvGuJm+1IXa0xAhhF3SOiHZDyNPLRC/QZ0x5lBOcOTZ6PkOltrM61AWui/LX/a40' + \
##      'fdx/gsLYLZbhQmB/qhU3xnk/FqMNGTx+5gplpxTtX3MX1OcUDePyY2p+XqWi0bWDAoGUXbiUyIyPi5' + \
##      'p2dpreYm4YIKgHn22+1DEoRA/TOk/RBkgF52TD3zx+h5AckIxwdkKM+PLyYkol0QDXxvLYPCMOpyKC' + \
##      'PgPppX65+AJXqF2W94A5L0I77yaA73es75XpeGarpqFLId3hKBrkGQ+5tmjui1ySb5OhprI0dWcg4g' + \
##      'w8OWJHDWq4lKtnOl8r5rP3b8YEaGoOZeyGRHPHET0T55GP+2CBKkQzB+zJicuMCnA8u1vZvchfy+kv' + \
##      'ItrrCxJkw/yXj1HBTbDtmCy6ltqknvSCh16Hnefmk6h6LRN76IS9x//Szvci4XV+B+tOJ/ujW1UDGn' + \
##      'baYBxtEHf9fxeMw+fMa7bXDTt7QUTKeN/+pIqRiUeBwtlgqbNFSlWyJz0sBxh76qHLhAtA/DtaO5Cc' + \
##      'V1BfgTdTaCD5R0d6Cd/hmP/DKDUNXptd4wGXKSlTA1qJfPN0GKGIyWiPuhnyuvqNtJhYJ/LJIUggBd' + \
##      'zwj50h/dT16F+MjPikA1NotOr/aZzXrMbrH84guenJTlgo9QJYW5s2+tVlBT5YXfU9QI7+NEecQhWS' + \
##      '0Acp7OjT1+JUE9ReO4D8uxtW2yfGHVjiRp4dbCI4XxJAnuMYRf5t9XyCDHP4u5nNtfFsJytC6mY/BD' + \
##      'SemfpDP7HG8sNVQWJG7eZ4KVEts3e+s4pIKrw8Ef2lXUI3nbMDqulMS/sj5/lX7cckSYcNBqCFwX4v' + \
##      'G2QJrLku1eepzs7pQwhKh/G7jrXzc2eRMGZbuKhSztQF8uTcfaKJSRKEpOU9rbaVMbnxkZSzv+Ci4q' + \
##      'p9zeh1tRqgBboAPU06oEsNzfVaJsCvvQYEzuhEjIZnwwIk+OGeYs+8hKUvfOr3H9gvSqD+wwGXoRyp' + \
##      'FFSiYDPZoLL7EM1Zl1mOk8hGJbZ9n62oK8pxanVoYTajiM0xy3jO9udA6odyklb8c4t0k+uaxpwrwz' + \
##      'FK2X1gAnB+wFWmxQZ74RBpXfcA1fFPB/JJ7jsQVpYdF5gZwK4ENDf7fmr8BklPTeWCD7cX7DPVgFb8' + \
##      'cd8Rmec0u/wYwoVHmvFV70JZaHmctMq/PXIjbMdy5/geaeBrVjUMxZ9UKcP6oBPM4jiL3vdaWdLlcI' + \
##      'Z8nnmbsbUP1yGr32rBg5431kq6n2wfyDFc7BlXrzGO6GvkfRM8V+Az3hml0/uM1rBmqv+KCecR/P02' + \
##      'yD9XznCepcxgSk6xibOD+4ieBt6FtMNfLvFYrrXQLRRkXt30iSVjGNbFhrNwYc0=',
##      'qshmxi95eNo=')) # generated by Delphi code
