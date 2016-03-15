# maya-json-export

[![experimental](http://badges.github.io/stability-badges/dist/experimental.svg)](http://github.com/badges/stability-badges)

[![demo](http://i.imgur.com/QBdTN1Q.png)](http://jam3.github.io/maya-json-exporter/index.html)

> [(demo)](http://jam3.github.io/maya-json-exporter/index.html)

A generic Maya to JSON exporter for triangulated meshes, specifically built for ThreeJS BufferGeometry. Designed for ThreeJS r74 and Maya 2016.

This is modified from [Sean Griffin's exporter](https://github.com/mrdoob/three.js/tree/master/utils/exporters/maya), but designed for static scenes with a lot of instancing (like a forest or city scape).

This assumes you will be unrolling your vertex data into un-indexed VBO. This was chosen because Maya, unlike WebGL, indexes UVs/Normals/Colors independently of position. See [here](https://github.com/mrdoob/three.js/issues/6926) for discussion.

Exports:

- Vertex positions and their indices
- Vertex normals and their indices
- Vertex UVs and their indices
- Material "Groups" for BufferGeometry
- Includes a crude de-duplication for generating instanced geometry
- Material data

> ###### :seedling: This plugin is experimental and subject to change.

## Demo

See the [demo](http://jam3.github.io/maya-json-exporter/index.html) which was exported with this plugin, and loaded directly into a `BufferGeometry`. 

## Export Format

The format roughly looks like this:

```json
{
  "metadata": {
    "exporter": "maya-json-export",
    "version": 0.0
  },
  "instances": [ 
    {
      "position": [0.0, 0.0, 0.0],
      "quaternion": [0.0, 0.0, 0.0, 1.0],
      "scale": [1.0, 1.0, 1.0],
      "id": "Tree",
      "name": "Tree_01"
    },
    {
      "position": [5.0, 2.0, 3.0],
      "quaternion": [0.0, 0.0, 0.0, 1.0],
      "scale": [1.0, 1.0, 1.0],
      "id": "Tree",
      "name": "Tree_02"
    },
    ... 
  ],
  "geometries": {
    "Tree": {
      "normal": [0, 0.5, 0, ...],
      "normalIndices": [ 0, 1, 2, ... ]
      "uv": [0, 0.33, ...],
      "uvIndices": [0, 1, 2, ...],,
      "position": [ 0, 5, 2, ... ],
      "positionIndices": [ 0, 1, 2, ... ],
      "groups": [
        { "start": 0, "count": 500, "materialIndex": 0 }
      ]
    }
  },
  "materials": [
    // like ThreeJS exporter
  ]
}
```

## Importers

See [demo/load.js](./demo/load.js) for an example of an importer with ThreeJS. When this plugin becomes more stable, the ThreeJS importer will be published to npm.

## Dedupe & Instancing

When the `Dedupe` option is enabled, the script will try to separate "geometries" from "instances" in the selected meshes. It does this by simply looking for the last underscore in the mesh name, and creating a dictionary of geometries by the first part of that.

For example, a city scene might have hundreds of skyscrapers repeated like so:

- `Building_Tall_Shape01`
- `Building_Tall_Shape02`
- `Building_Tall_Shape03`
- `Building_Short_Shape01`
- `Building_Short_Shape02`

With the regular ThreeJS exporter, you would end up with a lot of repeated triangles. Instead, with this exporter, you can deduplicate and end up with only two `geometries`, `Building_Tall` and `Building_Short`. The exporter provides 5 `instances`, each with their own attributes:

- `id` String (e.g. `'Building_Tall'`)
- `name` String (e.g. `'Building_Tall_Shape01'`)
- `position` [ x, y, z ]
- `scale` [ x, y, z ]
- `quaternion` [ x, y, z, w ]

Now, your scene will be much better optimized to make use of ThreeJS `Geometry` and `Mesh` classes. Or, you could use `InstancedBufferGeometry` to take advantage of hardware instancing.

## Install

Only tested on Maya 2016. You need PyMel (included with 2015+).

Copy the [exporter/](./exporter) files to your Maya `scripts` and `plug-ins` folder.

Example paths:

- Windows: `C:\Users\username\Documents\maya\VERSION\`
- OSX: `~/Library/Preferences/Autodesk/maya/VERSION/`
- Linux: `/usr/autodesk/userconfig/maya/VERSION/`

After that, you need to activate the plugin. In Maya, open `Windows > Settings/Preferences > Plug-in Manager` and enable the checkboxes next to `SimpleJSON.py`.

## License

MIT, see [LICENSE.md](http://github.com/Jam3/maya-json-export/blob/master/LICENSE.md) for details.
