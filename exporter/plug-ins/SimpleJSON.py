"""
A generic JSON mesh exporter for Maya.

Authors:
    Sean Griffin
    Matt DesLauriers
    Wijnand Koreman
    Roy Nieterau

"""

import sys
import logging
import os.path
import json
import pprint

import pymel.core as pm
import maya.OpenMaya as om
import maya.OpenMayaMPx as ommpx
import maya.api.OpenMaya as om2

kPluginTranslatorTypeName = 'SimpleJSON.json'
kOptionScript = 'SimpleJSONScript'
kDefaultOptionsString = ''


def _get_mdagpath(node):
    """Return maya.api.OpenMaya.MDagPath from Pymel node

    Args:
        node (pymel.core.PyNode): Maya node

    Returns:
        maya.api.OpenMaya.MDagPath: The path to the node.

    """
    sel = om2.MSelectionList()
    sel.add(node.name())
    return sel.getDagPath(0)


def is_triangulated(mesh):
    """Returns whether the mesh is fully triangulated.

    Args:
        mesh (pymel.nodetypes.Mesh): Mesh to check

    Returns:
        bool: Whether mesh is triangulated

    """
    fnMesh = om2.MFnMesh(_get_mdagpath(mesh))
    triangles_per_face, _ = fnMesh.getTriangles()
    return all(num == 1 for num in triangles_per_face)


def get_hierarchy(nodes):
    """Return the children hierarchy with all descendants.

    This is a workaround to `pm.listRelatives(allDescendents=True)` that
    will correctly include the instances present under the hierarchy, because
    `pm.listRelatives` does not support that correctly.

    Note: This not return the nodes itself, only the children.

    Args:
        nodes (list): List of nodes to get hierarchy from.

    Returns:
        list: List of pymel.core.PyNode nodes in the hierarchy.

    """
    result = set()
    children = pm.listRelatives(nodes, children=True, fullPath=True)
    while children:
        result.update(children)
        children = pm.listRelatives(children, children=True, fullPath=True)

    result.update(nodes)
    return list(result)


def get_meshes(selection=True):
    if not selection:
        return pm.ls(type="mesh",
                     allPaths=True,
                     shapes=True,
                     noIntermediate=True)

    # To support instanced shapes we can *not* just do
    # `pm.ls(dag=True, selection=True)` because it will
    # only return the first instance found or all (with `allPaths`)
    # including those outside of selection. So we get the
    # hierarchy ourselves.
    sel = pm.ls(sl=1)
    hierarchy = get_hierarchy(sel)
    return pm.ls(hierarchy,
                 type="mesh",
                 shapes=True,
                 noIntermediate=True)


class SimpleJSONWriter(object):
    """Exporter for ThreeJS JSON format optimized for buffergeometry."""
    def __init__(self):
        self.log = logging.getLogger(self.__class__.__name__)

        self.options = {
            "vertices": True,
            "normals": True,
            "uvs": True,
            "dedupe": True,
            "groups": False,
            "materials": False,
            "diffuseMaps": False,
            "specularMaps": False,
            "bumpMaps": False,
            "prettyOutput": True,
            "selectedOnly": True,
            "precisionTransform": 3,
            "precisionVertices": 3,
            "precisionNormals": 3,
            "precisionUv": 4,
            # When optimize instance matrix is True the identity position,
            # rotation and scale will not be included in the output data.
            # E.g. a scale of [1.0, 1.0, 1.0] will be removed from the data.
            "optimizeInstanceMatrix": False,
            # When `optimizeInstanceId` is enabled the `name` will not be
            # included with the instance data, only the `id` is stored.
            "optimizeInstanceId": False,
            # Whether to log verbosely to debug the output
            "verbose": False
        }

        self.materials = []
        self.geometries = {}
        self.instances = []

    def write(self, path):

        if self.options['verbose']:
            self.log.info("Using options: \n%s", pprint.pformat(self.options))

        # Collect the export data
        if self.options["materials"]:
            self.log.info("Exporting All Materials...")
            self._exportMaterials()
        self._exportMeshes()

        # Generate the JSON
        output = {
            'metadata': {
                'exporter': 'maya-json-export',
                'version': 0.0
            },
            'materials': self.materials,
            'instances': self.instances,
            'geometries': self.geometries
        }

        # If no materials are exported exclude the key from the data
        if not output['materials']:
            del output['materials']

        self.log.info("Collected %s geometries", len(self.geometries))
        self.log.info("Collected %s instances", len(self.instances))

        # if self.options['verbose']:
        #     self.log.debug("Exporting output: \n%s", pprint.pformat(output))

        self.log.info("Writing file...")
        # Write the file
        with file(path, 'w') as f:
            if self.options['prettyOutput']:
                f.write(json.dumps(output,
                                   sort_keys=True,
                                   indent=4,
                                   separators=(',', ': ')))
            else:
                f.write(json.dumps(output,
                                   separators=(",", ":")))

        self.log.info("Saved file to: %s", path)

    def _allMeshes(self):

        selection = self.options['selectedOnly']
        self.log.info("Export selected..." if selection else "Export all...")
        meshes = get_meshes(selection=selection)

        triangulated_meshes = []
        for mesh in meshes:
            if not is_triangulated(mesh):
                self.log.warning('Skipping %s since it is not triangulated',
                                 mesh.name())
                continue

            triangulated_meshes.append(mesh)

        self.log.info('Exporting %d meshes' % len(triangulated_meshes))
        return triangulated_meshes

    def _exportMeshes(self):

        dedupe = self.options['dedupe']
        for mesh in self._allMeshes():

            # Take the name from the transform so we don't get the exact
            # same name when the shape is instanced.
            transform = mesh.getParent()
            name = transform.nodeName()
            key = name.rsplit("_", 1)[0] if dedupe else name
            if not dedupe or key not in self.geometries:
                self._exportGeometry(mesh, key)
            else:
                self.log.debug('Repeating instance "%s" for: %s', key, name)

            self._exportMeshInstance(mesh, key, name)

    def _exportGeometry(self, mesh, key):
        self.log.info('Exporting geometry %s', mesh.name())

        # Get Python API 2.0 representations
        dagpath = _get_mdagpath(mesh)
        fn_mesh = om2.MFnMesh(dagpath)
        it_mesh_poly = om2.MItMeshPolygon(dagpath)

        # Generate the geometry data
        geom = {}
        if self.options["vertices"]:
            geom['position'] = self._getVertices(fn_mesh)
            geom['positionIndices'] = self._getFaces(it_mesh_poly)

        if self.options["normals"]:
            geom['normal'] = self._getNormals(fn_mesh)
            geom['normalIndices'] = self._getNormalIndices(fn_mesh)

        if self.options["uvs"]:
            geom['uv'] = self._getUVs(fn_mesh)
            geom['uvIndices'] = self._getUVIndices(it_mesh_poly)

        if self.options["groups"]:
            geom['groups'] = self._getGroups(mesh)

        self.geometries[key] = geom

    def _exportMeshInstance(self, mesh, geometryName, instanceName):

        parent = mesh.getParent()
        matrix = pm.xform(parent, query=True, worldSpace=True, matrix=True)

        transformation = om2.MTransformationMatrix(om2.MMatrix(matrix))
        translation = transformation.translation(om2.MSpace.kWorld)
        quaternion = transformation.rotation(asQuaternion=True)
        scale = transformation.scale(om2.MSpace.kWorld)

        # Warn the user when any shearing transformation is present so the
        # user will be aware the output might be slightly different from
        # what is present in the scene.
        shear = transformation.shear(om2.MSpace.kWorld)
        if any(shear):
            self.log.warning("Shear transformation is not supported: %s",
                             parent.name())

        precision = self.options['precisionTransform']
        instance = {
            'id': geometryName,
            'name': instanceName,
            'p': self._roundPos(translation, precision),
            's': [round(x, precision) for x in scale],
            'q': self._roundQuat(quaternion, precision)
        }

        # Optimize instance by removing redundant identity matrix data
        # that is a default transformation anyway. No need to store this
        # data in file format.
        if self.options['optimizeInstanceMatrix']:
            if instance['p'] == [0.0, 0.0, 0.0]:
                instance.pop("p")
            if instance['s'] == [1.0, 1.0, 1.0]:
                instance.pop("s")
            if instance['q'] == [0.0, 0.0, 0.0, 1.0]:
                instance.pop("q")

        if self.options['optimizeInstanceId']:
            instance.pop("name")

        self.instances.append(instance)

    def _roundPos(self, pos, precision):
        return map(lambda x: round(x, precision), [pos.x, pos.y, pos.z])

    def _roundQuat(self, rot, precision):
        return map(lambda x: round(x, precision), [rot.x, rot.y, rot.z, rot.w])

    def _getGroups(self, mesh):

        matIds = []
        groups = []
        numPoints = len(mesh.faces) * 3
        for face in mesh.faces:
            matIds.append(self._getMaterialIndex(face, mesh))

        # just one material index for whole geometry
        if all(x == matIds[0] for x in matIds):
            groups.append({'start': 0,
                           'count': numPoints,
                           'materialIndex': matIds[0]})

        # needs MultiMaterial
        else:
            lastId = matIds[0]
            start = 0
            for idx, matId in enumerate(matIds):
                if matId != lastId:
                    groups.append({'start': start * 3,
                                   'count': (idx - start) * 3,
                                   'materialIndex': lastId})
                    lastId = matId
                    start = idx
            # add final group
            groups.append({'start': start * 3,
                           'count': (len(mesh.faces) - start) * 3,
                           'materialIndex': lastId})

        return groups

    def _getMaterialIndex(self, face, mesh):
        if not hasattr(self, '_materialIndices'):
            self._materialIndices = dict([(mat['DbgName'], i) for i, mat
                                          in enumerate(self.materials)])

        if self.options['materials']:
            for engine in mesh.listConnections(type='shadingEngine'):
                if (pm.sets(engine, isMember=face) or
                        pm.sets(engine, isMember=mesh)):

                    for material in engine.listConnections(type='lambert'):
                        name = material.name()
                        if name in self._materialIndices:
                            return self._materialIndices[name]

        return -1

    def _getFaces(self, it_mesh_poly):
        """Return vertex indices per polygon.

        Args:
            it_mesh_poly (maya.api.OpenMaya.MItMeshPolygon): A polygon iterator

        Returns:
            list: The list of vertex indices for all faces

        """

        faces = []
        it_mesh_poly.reset()
        while not it_mesh_poly.isDone():
            faces += it_mesh_poly.getVertices()
            it_mesh_poly.next(1)

        return faces

    def _getVertices(self, fn_mesh):
        points = []
        precision = self.options['precisionVertices']
        for point in fn_mesh.getPoints(space=om2.MSpace.kObject):
            points += self._roundPos(point, precision)
        return points

    def _getNormals(self, fn_mesh):
        normals = []
        precision = self.options['precisionNormals']
        for normal in fn_mesh.getNormals(space=om2.MSpace.kObject):
            normals += self._roundPos(normal, precision)
        return normals

    def _getNormalIndices(self, fn_mesh):
        _, indices = fn_mesh.getNormalIds()
        indices = list(indices)
        return indices

    def _getUVIndices(self, it_mesh_poly):
        indices = []
        it_mesh_poly.reset()
        while not it_mesh_poly.isDone():
            for i in range(3):
                indices.append(it_mesh_poly.getUVIndex(i))
            it_mesh_poly.next(1)

        return indices

    def _getUVs(self, fn_mesh):
        uvs = []
        precision = self.options['precisionUv']
        for u, v in zip(*fn_mesh.getUVs()):
            uvs.append(round(u, precision))
            uvs.append(round(v, precision))
        return uvs

    def _exportMaterials(self):
        for mat in pm.ls(type='lambert'):
            self.materials.append(self._exportMaterial(mat))

    def _exportMaterial(self, mat):

        color_diffuse = map(lambda i: i * mat.getDiffuseCoeff(),
                            mat.getColor().rgb)
        result = {
            "DbgName": mat.name(),
            "blending": "NormalBlending",
            "colorDiffuse": color_diffuse,
            "depthTest": True,
            "depthWrite": True,
            "shading": mat.__class__.__name__,
            "opacity": mat.getTransparency().a,
            "transparent": mat.getTransparency().a != 1.0,
            "vertexColors": False
        }
        if isinstance(mat, pm.nodetypes.Phong):
            result["colorSpecular"] = mat.getSpecularColor().rgb
            result["reflectivity"] = mat.getReflectivity()
            result["specularCoef"] = mat.getCosPower()
            if self.options["specularMaps"]:
                self._exportSpecularMap(result, mat)
        if self.options["bumpMaps"]:
            self._exportBumpMap(result, mat)
        if self.options["diffuseMaps"]:
            self._exportDiffuseMap(result, mat)

        return result

    def _exportBumpMap(self, result, mat):
        for bump in mat.listConnections(type='bump2d'):
            for f in bump.listConnections(type='file'):
                result["mapNormalFactor"] = 1
                self._exportFile(result, f, "Normal")

    def _exportDiffuseMap(self, result, mat):
        for f in mat.attr('color').inputs():
            result["colorDiffuse"] = f.attr('defaultColor').get()
            self._exportFile(result, f, "Diffuse")

    def _exportSpecularMap(self, result, mat):
        for f in mat.attr('specularColor').inputs():
            result["colorSpecular"] = f.attr('defaultColor').get()
            self._exportFile(result, f, "Specular")

    def _exportFile(self, result, mapFile, mapType):
        src = mapFile.ftn.get()
        fName = os.path.basename(src)
        result["map{}".format(mapType)] = fName
        result["map{}Repeat".format(mapType)] = [1, 1]
        result["map{}Wrap".format(mapType)] = ["repeat", "repeat"]
        result["map{}Anisotropy".format(mapType)] = 4


class SimpleJSONTranslator(ommpx.MPxFileTranslator):
    """MPxFileTranslator for the ThreeJS JSON writer.

    This wraps the `SimpleJSONWriter` so that Maya can trigger it using the
    file menu's export command.

    """
    def __init__(self):
        ommpx.MPxFileTranslator.__init__(self)
        self.log = logging.getLogger(self.__class__.__name__)

    def haveWriteMethod(self):
        return True

    def filter(self):
        return '*.json'

    def defaultExtension(self):
        return 'json'

    def _parseOptions(self, optionsString):
        """Parse the options string into a dict"""

        options = dict()
        pairs = optionsString.split(";")
        for pair in pairs:
            if not pair or "=" not in pair:
                continue

            key, value = pair.split("=")
            options[key] = int(value)

        return options

    def writer(self, fileObject, optionString, accessMode):

        # Parse the options
        path = fileObject.fullName()
        options = self._parseOptions(optionString)

        if accessMode == ommpx.MPxFileTranslator.kExportAccessMode:
            selected_only = False
        elif accessMode == ommpx.MPxFileTranslator.kExportActiveAccessMode:
            selected_only = True
        else:
            raise ValueError("Unsupported accessMode: {0}".format(accessMode))

        # Write
        exporter = SimpleJSONWriter()
        exporter.options.update(options)
        exporter.options['selectedOnly'] = selected_only
        exporter.write(path)


def translatorCreator():
    return ommpx.asMPxPtr(SimpleJSONTranslator())


def initializePlugin(mobject):
    mplugin = om.MFnPlugin(mobject)
    try:
        mplugin.registerFileTranslator(kPluginTranslatorTypeName,
                                       None,
                                       translatorCreator,
                                       kOptionScript,
                                       kDefaultOptionsString)
    except:
        sys.stderr.write('Failed to register translator: '
                         '%s' % kPluginTranslatorTypeName)
        raise


def uninitializePlugin(mobject):
    mplugin = om.MFnPlugin(mobject)
    try:
        mplugin.deregisterFileTranslator(kPluginTranslatorTypeName)
    except:
        sys.stderr.write('Failed to deregister translator: '
                         '%s' % kPluginTranslatorTypeName)
        raise
