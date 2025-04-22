## Watcher Teleportation Overview

![overview](./readme.assets/overview.svg)

## Features

* Automatically arranges map positions based on algorithms
* Rendering support
  * Sand (`TerrainHandle`)
  * Wall-adhering fungus climbable long tentacles (`CorruptionTube`)
  * Acid fluids (`LethalWater` and `Toxic Brine Water`)
  * Air zones in water (`AirPocket`, `WaterCutoff`)
  * Floating water levels (`WaterFluxMinLevel`, `WaterFluxMaxLevel`, `WaterCycleTop`, `WaterCycleBottom`)
* Region support
  * Observer's industrial subregion `Hydroponics`
* Annotations
  * Echo locations
  * Teleport locations
  * Pearl and Karma Flower locations
  * Room types (e.g., `Shelter`)
  * Room numbers

## Notes

* In Watcher, the search order of World files is likely `watcher`, `vanilla`, `moreslugcat`.
* When teleporting from a corrupted vanilla world fissure to the outer expanse, the arrival room is random; `WORA_START` is only one of the possible rooms.
  * The destination room to the Demon region also appears to be random

### TODO

- [ ] Echo teleport location from `WAUA_BATH` to `WAUA_TOYS` is slightly inaccurate

## Links

### Map Generator

* Map editor with UI — [Cornifer](https://github.com/Ved-s/Cornifer)
* [Cornifer](https://github.com/Ved-s/Cornifer) branch adapted for Watcher — [branch](https://github.com/enchanted-sword/Cornifer)

### Map Format Documentation

* [Creating A Region - Rain World Modding](https://rainworldmodding.miraheze.org/wiki/Creating_A_Region)
* [World File Format - Rain World Modding](https://rainworldmodding.miraheze.org/wiki/World_File_Format)
* [Level Editor - Rain World Modding](https://rainworldmodding.miraheze.org/wiki/Level_Editor)

## Reference Effects

### Layout Algorithm Animation

<img src="./readme.assets/anima.gif" alt="anima" style="zoom:50%;" /><img src="./readme.assets/ward.png" alt="Cold Storage (WARD)" style="zoom: 10%;" />

### Sand Rendering

![image-20250420173432820](./readme.assets/image-20250420173432820.png)

### Air Zones in Water

<img src="./readme.assets/image-20250420173504926.png" alt="image-20250420173504926" style="zoom:33%;" /><img src="./readme.assets/image-20250420173522096.png" alt="image-20250420173522096" style="zoom:33%;" />

### Climbable Long Tentacles for Wall-Adhering Fungus

<img src="./readme.assets/image-20250420173610059.png" alt="image-20250420173610059" style="zoom:33%;" />

### Floating Water Levels

![image-20250420173815503](./readme.assets/image-20250420173815503.png)

### Industrial Complex

![Industrial Complex (HI)](./readme.assets/hi.png)

### Salination Region

![Salination (WARB)](./readme.assets/warb.png)