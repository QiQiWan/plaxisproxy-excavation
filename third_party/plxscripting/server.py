"""
Purpose: the Server provides proxy clients with methods for manipulating and
    querying the Plaxis environment and its global objects.

    The methods construct strings and call the Plaxis local server.

    The subsequent output is processed to create data for proxy client objects.
    This could take the form of a list of GUIDs or it could take the form of
    a string if requesting information about the state of the environment. If
    the request is not sucessful then a scripting exception is raised with
    the message that was returned from the interpreter.

Subversion data:
    $Id: server.py 20240 2015-09-22 13:24:27Z tj $
    $URL: https://tools.plaxis.com/svn/sharelib/trunk/PlxObjectLayer/Server/plxscripting/server.py $

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

import sys
import keyword
from re import search

from .plx_scripting_exceptions import PlxScriptingError, PlxScriptingLocalError
from .plxproxyfactory import is_primitive, TYPE_OBJECT, PlxProxyFactory
from .connection import HTTPConnection
from .image import TYPE_NAME_IMAGE, create_image
from .const import (PLX_CMD_NEW, PLX_CMD_CLOSE, PLX_CMD_OPEN, PLX_CMD_RECOVER, TOKENIZE,
                    JSON_COMMANDS, JSON_FEEDBACK, JSON_SUCCESS, JSON_EXTRAINFO,
                    JSON_GUID, JSON_TYPE, JSON_RETURNED_OBJECTS, JSON_RETURNED_VALUES,
                    JSON_PROPERTIES, JSON_QUERIES, JSON_NAMEDOBJECTS, JSON_MEMBERNAMES,
                    JSON_RETURNED_OBJECT, JSON_OWNERGUID, JSON_ISLISTABLE,
                    JSON_TYPE_JSON, JSON_KEY_JSON, JSON_KEY_CONTENT_TYPE, JSON_VALUE,
                    METHOD, GUID, COUNT, STAGED_PREFIX,
                    JSON_LISTQUERIES, JSON_METHODNAME, JSON_OUTPUTDATA, JSON_SELECTION,
                    SUBLIST, INDEX, MEMBERSUBLIST, MEMBERINDEX, STARTINDEX, STOPINDEX, MEMBERNAMES,
                    NULL_GUID, LOCAL_HOST,
                    JSON_NAME, PLAXIS_2D, PLAXIS_3D, ARG_APP_SERVER_ADDRESS, ARG_APP_SERVER_PORT, ARG_PASSWORD)
from .selection import Selection
from .logger import Logger
from .error_mode import ErrorMode
from .tokenizer import TokenizerResultHandler
from types import GeneratorType

try:
    basestring  # will give NameError on Py3.x, but exists on Py2.x (ancestor of str and unicode)


    def is_str(s):
        return isinstance(s, basestring)
except NameError:
    def is_str(s):  # Py3.x only has one type of string
        return isinstance(s, str)

plx_string_wrappers = ['"', "'", '"""', "'''"]


class InputProcessor(object):
    """
    Helper class which processes scripting input in order to present it
    correctly to the Plaxis server.
    """

    def param_to_string(self, param):
        try:
            return param.get_cmd_line_repr()
        except AttributeError:  # param has no appropriate method -> maybe it's a primitive
            # strings need to wrapped with quotes (depending on whether they contain quotes themselves)
            if is_str(param):
                for wrapper in plx_string_wrappers:
                    if not wrapper in param:
                        return wrapper + param + wrapper
                raise PlxScriptingLocalError("Cannot convert string parameter to valid Plaxis string"
                                             " representation, try removing some quotes: " + param)
            elif isinstance(param, (tuple, list)):  # tuples/lists wrapped with parens
                return '(' + self.params_to_string(param) + ')'
            elif isinstance(param, Selection):
                return self.param_to_string(list(param))
            elif isinstance(param, GeneratorType):  # expand generator contents (it gets consumed)
                return self.param_to_string(list(param))
            else:
                return str(param)

    def params_to_string(self, params):
        """
        Takes a sequence and concatenates its contents into a single
        space separated string.
        E.g.
            params: (1, 2, 3)
            returns "1 2 3"

            params: ()
            returns ""
        """
        # TODO handle more complex cases such as the following:
        #   * methods as params (as found when using the 'map' command)
        return " ".join([self.param_to_string(p) for p in params])

    def create_method_call_cmd(self, target_object, method_name, params):
        """
        Arranges the command line name of a proxy object, method name and
        parameters into a string with a format matching the Plaxis command
        line.
        E.g.
            target_obj_name: 'Point_1'
            method_name: "move"
            params: (3, 4, 5)
            returns "move Point_1 3 4 5"

            target_obj_name: "" (in the case of the global object)
            method_name: "undo"
            params: ()
            returns "undo"
        """
        param_string = self.params_to_string(params)

        parts = [method_name]
        if target_object is not None:
            target_cmd_line_repr = target_object.get_cmd_line_repr()
            if target_cmd_line_repr:
                parts.append(target_cmd_line_repr)
        if param_string:
            parts.append(param_string)
        return " ".join(parts)


class ResultHandler(object):
    """
    Helper class which parses the output of the Plaxis server and returns
    objects, primitives, booleans or strings if successful. Otherwise an
    exception is raised.
    """

    def __init__(self, server, proxy_factory):
        self.proxy_factory = proxy_factory
        self.server = server
        self._json_constructors = {}
        self._last_response = ""

    @property
    def last_response(self):
        return self._last_response

    def register_json_constructor(self, name, factory_function):
        self._json_constructors[name] = factory_function

    def _handle_successful_command(self, commands_response):
        """
        Supplied with a response from a successful command, returns one of
        the following if they are present: a list of proxy objects, extra
        information from the command line or True as a fallback.
        """
        obj_list = self._create_proxies_from_returned_objects(
            commands_response[JSON_RETURNED_OBJECTS])

        if obj_list is not None:
            return obj_list
        elif len(commands_response.get(JSON_RETURNED_VALUES, [])) > 0:
            return commands_response[JSON_RETURNED_VALUES]
        else:
            json_extra_info = commands_response[JSON_EXTRAINFO]
            if json_extra_info:
                return json_extra_info
        return True

    def handle_namedobjects_response(self, namedobjects_response):
        """
        Handles the JSON response to a call to the namedobjects resource.
        May return some newly created objects or a scripting error if the
        named object is not present.
        """
        is_namedobject_successful = namedobjects_response[JSON_SUCCESS]
        self._last_response = namedobjects_response[JSON_EXTRAINFO]

        if is_namedobject_successful:
            return self._create_proxies_from_returned_objects(
                [namedobjects_response[JSON_RETURNED_OBJECT]])
        else:
            raise PlxScriptingError("Unsuccessful command:\n" +
                                    namedobjects_response[JSON_EXTRAINFO])

    def handle_commands_response(self, commands_response):
        """
        Handles the (JSON) response to a Plaxis method call. May return some
        newly created objects or some text indicating some change of state. If
        the method call is not successful, then an exception is raised.
        """
        is_command_successful = commands_response[JSON_SUCCESS]
        self._last_response = commands_response[JSON_EXTRAINFO]

        if is_command_successful:
            return self._handle_successful_command(commands_response)
        else:
            raise PlxScriptingError("Unsuccessful command:\n" +
                                    commands_response[JSON_EXTRAINFO])

    def handle_members_response(self, members_response, proxy_obj):
        """
        Constructs and returns a dictionary containing the attribute names
        of an object mapped to the relevant proxy entity (either a proxy method
        or a proxy property). The supplied membernames response is the JSON
        object from the server that represents the attributes of the object.
        """
        proxy_attributes = {}

        if JSON_COMMANDS in members_response:
            commands_list = members_response[JSON_COMMANDS]
            for method_name in commands_list:
                # Remove the __ bit from method names, because PlxProxyObjects
                # use __getattr__ to access the methods/properties, the method
                # name would be changed by Python's name mangling.
                # Also append a _ when the method name conflicts with a Python
                # keyword.
                exposed_name = method_name
                if exposed_name.startswith('__'):
                    exposed_name = exposed_name[2:]

                if keyword.iskeyword(exposed_name):
                    exposed_name = exposed_name + '_'

                proxy_method = self.proxy_factory.create_plx_proxy_object_method(
                    self.server, proxy_obj, method_name)
                proxy_attributes[exposed_name] = proxy_method

        if JSON_PROPERTIES in members_response:
            properties_dict = members_response[JSON_PROPERTIES]

            for property_name in sorted(properties_dict.keys()):
                ip = self._create_proxy_object(
                    properties_dict[property_name], property_name, proxy_obj)
                proxy_attributes[property_name] = ip

        return proxy_attributes

    def handle_list_response(self, list_response):
        """
        Handles the response to a call to the list resource. Depending on the
        call and the state of the project, the response may be a primitive,
        a proxy object, a list of proxy objects, or an error.
        """
        is_listquery_successful = list_response[JSON_SUCCESS]
        if is_listquery_successful:
            method_name = list_response[JSON_METHODNAME]
            output_data = list_response[JSON_OUTPUTDATA]
            if method_name == COUNT:
                return output_data
            elif method_name == SUBLIST:
                # Sublists (even if just one item large) should still be regarded as lists,
                # otherwise asking for e.g. g.Lines[:] when there is just one line will
                # return either a line object directly or a list of line objects. This
                # makes it rather hard to write code using list slices, as you never
                # know what to expect out of them.
                return self._create_proxies_from_returned_objects(output_data,
                                                                  allow_one_item_list_result=True)
            elif method_name == INDEX:
                return self._create_proxies_from_returned_objects(
                    [output_data])
            elif method_name == MEMBERSUBLIST:
                # We assume that a single member name has been queried for now
                queried_member = list_response[JSON_MEMBERNAMES][0]
                return self._create_proxies_from_returned_objects(output_data[queried_member],
                                                                  allow_one_item_list_result=True)
            elif method_name == MEMBERINDEX:
                # We assume that a single member name has been queried for now
                queried_member = list_response[JSON_MEMBERNAMES][0]
                return self._create_proxies_from_returned_objects(
                    [output_data[queried_member]])

        raise PlxScriptingError("Unsuccessful command:\n" +
                                list_response[JSON_EXTRAINFO])

    def handle_propertyvalues_response(self, propertyvalues_response_list,
                                       attr_name, owner_type):
        """
        Handle the request for a list of properties. Returns
        a list of properties. If there is no such attribute for one
        of the queries, then the method returns None.
        """
        response = []
        for single_property_json in propertyvalues_response_list:
            single_property_response = self._get_single_propertyvalues(single_property_json, attr_name, owner_type)
            response.append(single_property_response)

        return response

    def _get_single_propertyvalues(self, single_property_json, attr_name, owner_type):
        """
        Handle the request for a property. Returns the property.
        If there is no such attribute then the method returns None.
        """
        # Make a call to Plaxis to get the object's properties
        if JSON_PROPERTIES in single_property_json:
            property_names = single_property_json[JSON_PROPERTIES]
            if attr_name in property_names:
                attribute = property_names[attr_name]
                # TODO: work out how to "proxify" non-staged primitives considering the ones in UserFeatures
                if isinstance(attribute, dict):
                    # Staged intrinsic properties are different from normal
                    # intrinsic properties because their owner is an object
                    # that isn't accessible from the scripting layer. This is
                    # why they are returned as a dict so the proxies can be
                    # built in a different way.
                    if owner_type.startswith(STAGED_PREFIX):
                        if is_primitive(attribute[JSON_TYPE]):
                            return self._create_stagedIP_proxy(attribute, attr_name)
                        elif attribute[JSON_TYPE] == TYPE_OBJECT:
                            if attribute[JSON_VALUE] != NULL_GUID:
                                return self._create_proxy_object(attribute[JSON_VALUE])
                            else:
                                return None

                    return self._create_proxies_from_returned_objects([attribute])
                elif attribute == NULL_GUID:
                    return None
                return attribute

        return None

    def handle_selection_response(self, selection_response):
        selection_objects = selection_response[JSON_SELECTION]
        result = self._create_proxies_from_returned_objects(selection_objects,
                                                            allow_one_item_list_result=True)

        if result is None:  # No selected objects is perfectly valid.
            return []
        else:
            return result

    def _create_stagedIP_proxy(self, returned_object, attr_name):
        """"Creates a proxy for the staged IP primitive values"""
        guid = returned_object[JSON_GUID]
        primitive_type = returned_object[JSON_TYPE]
        # The owner of stagedIP's are not set.
        primitive_proxy = self.proxy_factory._create_plx_proxy_property(self.server, guid, primitive_type, False,
                                                                        attr_name, None)
        # The value is set here since their owner is None
        primitive_proxy.set_stagedIP_value(returned_object[JSON_VALUE])
        return primitive_proxy

    def _create_proxy_object(self, returned_object, prop_name=None, owner=None):
        """
        Accesses the data for a returned object and creates a proxy from that
        data.
        """
        # JSON payload doesn't need a proxy object so just return the dict.
        # Unless it contains the 'type' property, then try to make an object
        # out of it.
        if returned_object[JSON_TYPE] == JSON_TYPE_JSON:
            json_object = returned_object[JSON_KEY_JSON]
            if isinstance(json_object, dict) and JSON_KEY_CONTENT_TYPE in json_object:
                constructor_name = json_object[JSON_KEY_CONTENT_TYPE]
                constructor = self._json_constructors.get(constructor_name)
                if constructor is None:
                    raise Exception("Constructor {} is not registered.".format(
                        constructor_name))

                return constructor(json_object)

            return json_object

        guid = returned_object[JSON_GUID]

        # cast to str needed for Py2.7, where otherwise a potential unicode-type result object could lead to problems
        plx_obj_type = str(returned_object[JSON_TYPE])

        is_listable = returned_object[JSON_ISLISTABLE]

        # If we approach an object as listable, but its listification actually
        # returns intrinsic property objects that have NOT YET been cached as
        # proxies, we still need to identify them as intrprops and create the
        # appropriate proxy objects for them.
        if owner is None:  # cannot simply write "if not owner", because proxies may implement __bool__ and return False even if the owner does exist
            if JSON_OWNERGUID in returned_object:
                owner = self.proxy_factory.get_proxy_object_if_exists(returned_object[JSON_OWNERGUID])
                # It shouldn't be possible to get a property back *before* we have
                # instantiated a proxy object for that property's owner.
                if owner is None:  # also here cannot simply write "if not owner" for similar reasons as above
                    raise PlxScriptingError('Missing owner object for property object!')
                    # The fact that the owner exists, doesn't necessarily mean all
                # its intrinsic properties have been retrieved too; and we need
                # them retrieved, because in the problematic situation we're
                # dealing with here, prop_name is *also* unknown and we can
                # therefore not instantiate the property ourselves!
                if self.proxy_factory.get_proxy_object_if_exists(guid) is None:
                    # also here cannot simply write "if not self.proxy_factory.get_proxy_object_if_exists(...)" for similar reasons as above
                    self.server.get_object_attributes(owner)

        return self.proxy_factory.create_plx_proxy_object(
            self.server, guid, plx_obj_type, is_listable, prop_name, owner)

    def _create_proxies_from_returned_objects(self, returned_objects, allow_one_item_list_result=False):
        """
        Given a returned objects list from the API, creates relevant proxy
        objects for each returned object representation. If the list contains
        just one object representation, a single proxy is returned. If the
        list contains more than one object representation, a list of proxies
        is returned.
        If allow_one_item_list_result==False and the returned object list contains
        just one item, this method will return just that one item on its own
        (i.e. not wrapped in a list). If the parameter is True, it will return
        it a one-item list.
        """
        new_objs = []

        for returned_object in returned_objects:
            if isinstance(returned_object, dict):
                obj = self._create_proxy_object(returned_object)
            else:
                obj = returned_object
            new_objs.append(obj)

        if len(new_objs) == 1 and not allow_one_item_list_result:
            return new_objs[0]

        if new_objs == []:
            # If we simply return the empty-list and the caller wants to check
            # whether some item was returned, it's tempting to write "if result: ...".
            # This will however go wrong if we actually return new_objs[0] (see above)
            # and that object happens to evaluate to False in a boolean context.
            # Therefore we need to distinguish clearly between NO results and some
            # result that may evaluate to False in a boolean context.
            return None
        else:
            return new_objs


class Server(object):
    """
    Provides proxy clients with the means to request and receive
    information from a connection to Plaxis.
    """

    def __init__(self, connection, proxy_factory, input_processor, allow_caching=True):
        """
        If values and global objects are cached, this reduces the number of calls to
        the server, but if the project changes outside this scripting environment,
        the internal state will be invalid.
        The caching system is reset whenever a call is made to the server that *might*
        change values or global objects. Obviously we don't know what side-effects
        commands and such have, so all commands are regarded as cache-invalidating.
        """
        self.connection = connection
        self.input_proc = input_processor
        self.__allow_caching = allow_caching
        self._proxies_to_reset = []
        self.reset_caches()
        self._server_name = None
        self.error_mode = connection.error_mode

        # TODO unsure about the tight coupling here
        self.plx_global = proxy_factory.create_plx_proxy_global(self)
        self.result_handler = ResultHandler(self, proxy_factory)
        self.__proxy_factory = proxy_factory

    @property
    def allow_caching(self):
        return self.__allow_caching

    @allow_caching.setter
    def allow_caching(self, value):
        if self.__allow_caching != value:
            self.__allow_caching = value
            self.reset_caches()

    @property
    def active(self):
        return self.connection.poll_connection()

    @property
    def last_response(self):
        return self.result_handler.last_response

    @property
    def major_version(self):
        matches = search(r'(\d*)\.(\d*).(\d*)\.(\d*)', self.server_full_name)
        return int(matches.group(1))

    @property
    def minor_version(self):
        matches = search(r'(\d*)\.(\d*).(\d*)\.(\d*)', self.server_full_name)
        return int(matches.group(2))

    @property
    def name(self):
        if 'PLAXIS 3D' in self.server_full_name:
            return PLAXIS_3D
        if 'PLAXIS 2D' in self.server_full_name:
            return PLAXIS_2D

    @property
    def server_full_name(self):
        if not self._server_name:
            self._server_name = self.connection.request_server_name()
        return self._server_name

    @property
    def is_2d(self):
        return self.name == PLAXIS_2D

    @property
    def is_3d(self):
        return self.name == PLAXIS_3D

    def enable_logging(self, **kwargs):
        """
        Enables the logging of requests made to the server. If no arguments are
        given, a file name will be generated in the %TEMP%/PlaxisScriptLogs
        directory.

        Args:
          file: if specified, file object to which to log (opened for writing)
          path: if specified, file name to which to log (must have write access)
        """
        self.connection.logger = Logger(**kwargs)

    def disable_logging(self):
        self.connection.logger = None

    def add_proxy_to_reset(self, proxy_obj):
        self._proxies_to_reset.append(proxy_obj)

    def reset_caches(self):
        for proxy_obj in self._proxies_to_reset:
            proxy_obj.reset_cache()

        self.__globals_cache = {}
        self.__values_cache = {}
        self.__listables_cache = {}

    def new(self):
        """Create a new project """
        result = self.connection.request_environment(PLX_CMD_NEW)
        if result:
            self.reset_caches()
            self.__proxy_factory.clear_proxy_object_cache()
        return result

    def recover(self):
        """Recover a project """
        result = self.connection.request_environment(PLX_CMD_RECOVER)
        if result:
            self.reset_caches()
            self.__proxy_factory.clear_proxy_object_cache()
        return result

    def open(self, filename):
        """Open a project with the supplied name """
        result = self.connection.request_environment(PLX_CMD_OPEN, filename)
        if result:
            self.reset_caches()
            self.__proxy_factory.clear_proxy_object_cache()
        return result

    def close(self):
        """Close the current project """
        result = self.connection.request_environment(PLX_CMD_CLOSE)
        if result:
            self.reset_caches()
            self.__proxy_factory.clear_proxy_object_cache()
        return result

    def __get_with_cache(self, key, cache, func_if_not_found):
        """
        Utility function that can be used to abstract away the behaviour
        of the different caches. It receives the lookup key for the
        cache, the cache object and the function to call if the key
        is not found in the cache (or if caching is disabled).
        """
        if self.allow_caching and key in cache:
            return cache[key]

        result = func_if_not_found()

        if self.allow_caching:
            cache[key] = result

        return result

    def __call_listable_method_no_cache(self, proxy_listable, method_name, startindex, stopindex, property_name):
        optional_parameters = {}

        # Attach any supplied index arguments to the query
        if startindex is not None:
            optional_parameters[STARTINDEX] = startindex
        if stopindex is not None:
            optional_parameters[STOPINDEX] = stopindex
        if property_name is not None:
            optional_parameters[MEMBERNAMES] = [property_name]

        listable_query = {GUID: proxy_listable._guid, METHOD: method_name}
        listable_query.update(optional_parameters)

        response = self.connection.request_list(listable_query)
        return self.result_handler.handle_list_response(
            response[JSON_LISTQUERIES][0])

    def call_listable_method(self, proxy_listable, method_name, startindex=None,
                             stopindex=None, property_name=None):
        """
        Constructs a listable query and returns the handled response.
        """
        key = (proxy_listable._guid, method_name, startindex, stopindex, property_name)
        return self.__get_with_cache(key, self.__listables_cache,
                                     lambda: self.__call_listable_method_no_cache(
                                         proxy_listable, method_name, startindex, stopindex, property_name))

    def __get_name_object_no_cache(self, object_name):
        response = self.connection.request_namedobjects(object_name)
        return self.result_handler.handle_namedobjects_response(
            response[JSON_NAMEDOBJECTS][object_name])

    def get_named_object(self, object_name):
        """
        Return a representation of the named object.
        """
        return self.__get_with_cache(object_name, self.__globals_cache,
                                     lambda: self.__get_name_object_no_cache(object_name))

    def __get_objects_property_no_cache(self, proxy_objects, prop_name, phase_object):
        if phase_object:
            response = self.connection.request_propertyvalues(
                [po._guid for po in proxy_objects], prop_name, phase_object._guid)
        else:
            response = self.connection.request_propertyvalues(
                [po._guid for po in proxy_objects], prop_name)

        # handle a legacy signature response that is used in case of a single object request
        response_list = [response[JSON_QUERIES][proxy_objects[0]._guid]] if len(proxy_objects) == 1 else [
            response[JSON_QUERIES][i][po._guid] for (i, po) in enumerate(proxy_objects)]

        return self.result_handler.handle_propertyvalues_response(
            response_list, prop_name, proxy_objects[0]._plx_type)

    def get_objects_property(self, proxy_objects, prop_name, phase_object=None):
        """
        Gets the specified property value for a list of proxy objects.
        """
        key = (' '.join([po._guid for po in proxy_objects]), prop_name, phase_object)
        return self.__get_with_cache(key, self.__values_cache,
                                     lambda: self.__get_objects_property_no_cache(proxy_objects, prop_name,
                                                                                  phase_object))

    def get_object_property(self, proxy_object, prop_name, phase_object=None):
        """
        Gets the specified property value for the specified proxy object.
        """
        return self.get_objects_property([proxy_object], prop_name, phase_object)[0]

    def set_object_property(self, proxy_property, prop_value):
        """
        Sets the specified property value for the specified proxy object.
        """
        self.reset_caches()
        return self.call_plx_object_method(proxy_property,
                                           'set', prop_value)

    def get_object_attributes(self, proxy_obj):
        """
        Create a dictionary of object attributes mapped to their proxy
        equivalents (proxy methods or proxy properties)
        """
        response = self.connection.request_members(proxy_obj._guid)
        return self.result_handler.handle_members_response(
            response[JSON_QUERIES][proxy_obj._guid], proxy_obj)

    def call_plx_object_method(self, proxy_obj, method_name, params):
        """
        Calls a Plaxis method using the supplied proxy object, method name and
        parameters. Returns new objects, success infomation, or a boolean if
        the command succeeds. Otherwise a scripting error is raised with any
        error information.
        E.g.
            proxy_obj: a Point object
            method_name: "move"
            params: (1, 2, 3)
            returns "OK" (from Plaxis command line)
        E.g.
            proxy_obj: the global proxy object
            method_name: "line"
            params: (1, 1, 1, 0, 0, 0)
            returns a list of PlxProxyObjects
        """
        self.reset_caches()
        method_call_cmd = self.input_proc.create_method_call_cmd(
            proxy_obj, method_name, params)

        return self.call_and_handle_command(method_call_cmd)

    def call_and_handle_command(self, command):
        """
        Helper method which sends the supplied command string to the commands
        resource. Returns the handled response to that command.
        """
        response = self.call_and_handle_commands(command)
        return response[0] if len(response) > 0 else None

    def call_and_handle_commands(self, *commands):
        """
        Helper method which sends the supplied command string to the commands
        resource. Returns the handled response to that command.
        """
        response = self.call_commands(*commands)
        return [
            self.result_handler.handle_commands_response(r[JSON_FEEDBACK])
            for r in response]

    def call_commands(self, *commands):
        """
        Helper method which sends the supplied command string to the commands
        resource. Returns the handled response to that command or False.
        """
        self.reset_caches()
        response = self.connection.request_commands(*commands)
        return response.get(JSON_COMMANDS, [])

    def call_selection_command(self, command, *objects):
        """
        Changes the selection and returns the resulting selection afterwards.
        """
        guids = (o._guid for o in objects)
        response = self.connection.request_selection(command, *guids)
        return self.result_handler.handle_selection_response(response)

    def get_name_by_guid(self, guid):
        response = self.connection.request_propertyvalues([guid], JSON_NAME)
        return response[JSON_QUERIES][guid][JSON_PROPERTIES][JSON_NAME]

    def get_error(self, clear=True):
        return self.connection.request_exceptions(clear)

    def tokenize(self, command):
        response = self.connection.request_tokenizer([command])
        tokenize_list = response.get(TOKENIZE, [])
        return TokenizerResultHandler(tokenize_list[-1])


def _get_argument(arg_name):
    for arg in sys.argv:
        if arg_name.lower() in arg.lower():
            ignore, value = arg.split('=', 1)
            return value
    raise Exception("Couldn't get {} from command line arguments.".format(arg_name))


def new_server(address=None, port=None, timeout=5.0, request_timeout=None, password=None, error_mode=()):
    ip = InputProcessor()

    if address is None:
        try:
            address = _get_argument(ARG_APP_SERVER_ADDRESS)
        except Exception:
            address = LOCAL_HOST

    if port is None:
        try:
            port = int(_get_argument(ARG_APP_SERVER_PORT))
        except:
            port = 10000

    if password is None:
        password = _get_argument(ARG_PASSWORD)

    error_mode = ErrorMode(*error_mode)

    conn = HTTPConnection(address, port, timeout, request_timeout, password, error_mode=error_mode)
    pf = PlxProxyFactory(conn)
    s = Server(conn, pf, ip)

    s.result_handler.register_json_constructor(TYPE_NAME_IMAGE, create_image)

    return s, s.plx_global
