# PolarBench

PolarBench is a local inspection workbench for four-angle linear-polarization images of coated surfaces, reflective material and fiber laminates. A capture keeps its original analyzer measurements, calibration references, detector-space regions and saved inspections together.

## Run locally

```sh
docker compose up --build -d
```

Open `http://127.0.0.1:18943/`. The Compose volume holds captures, references, regions and views. `docker compose down` stops the service and keeps the volume. Set `POLARBENCH_PORT` to change the host port.

For Python execution, create a virtual environment using Python 3.11 or 3.12, install `requirements.txt` and run `./start.sh`. In this mode `POLARBENCH_DATA_DIR` defaults to `.local-data`, the service binds to loopback and `PORT` defaults to `18943`.

## Walk through an inspection

1. Choose **Open coated panel** for a synthetic four-angle capture. In **Map**, switch from **Linear degree** to **Polarization direction** or **Virtual analyzer**. Move the virtual analyzer angle, or click inside the visible central coverage gap to check the native alpha evidence.
2. In **Measure**, choose **Native coverage gap** and click **Measure region**. To read across the same area, draw with **Line** and click **Plot scan line**. Interpolation stops across gaps instead of marking excluded positions as zero.
3. In **Quality**, open the quality map, then locate **Native coverage gap** patches. Draw a box and preview an exclusion before applying it to the detector inclusion mask.
4. Use **Save view** to preserve a processing and source snapshot. **Export** offers a rendered PNG, linear numeric TIFF, inspection worksheet, contact sheet or evidence ZIP.

## Supported workflows

| Workflow | Entry and result |
| --- | --- |
| Four-angle capture | Import 0°, 45°, 90° and 135° grayscale images. Source download returns the original bytes. |
| Packed detector import | Import an even-sized 2 × 2 micro-polarizer mosaic with a known parity order. **Sources → Sensor tiles** shows the named detector lanes. |
| Inter-frame registration | Use the **Shifted laminate** demo or your own sequential capture. Suggest or set integer shifts relative to 0°, then review the overlay and texture agreement. |
| Intensity calibration | Upload dark and flat frames, choose a flat normalization window, inspect reference evidence and apply reference correction or per-angle black/gain. |
| Stokes reconstruction | Inspect intensity, Q/U, polarized intensity, degree and orientation from common valid support. Change the angular zero against a known axis. |
| Virtual analyzer | Review one analyzer angle, compare theoretical per-pixel bounds or examine a region through a sweep of angles. |
| Coverage diagnosis | Use categorical reasons, connected issue patches and opposite-pair evidence. Probe a detector pixel for its native source coordinates and available measurements. |
| Detector mask | Import positive grayscale inclusion, review interior and edge exclusions, or preview and apply a box edit. |
| Regional measurement | Measure saved or live regions, inspect a value histogram and axial summary and export regional statistics. |
| Scan-line sampling | Draw or save a detector line and sample with nearest or bilinear interpolation while preserving invalid intervals. |
| Spatial detector summaries | Read orientation by grid cell, row bands or signal extrema. Calibrate detector pitch to report region dimensions or line lengths in physical units. |
| Raw crop | Preview a box and save it as a new capture with unchanged original sample codes, complete mosaic tiles and contained regions. |
| Capture comparison | Compare same-size detector captures, review support and signed change and download numerical comparison evidence. |
| Saved inspection | Replay a previous view using the original mask/references or restore its controls while the current source identity matches. |
| Evidence and portable source packages | Export maps, scalar maps and support together or move original data, current settings and regions into another PolarBench workspace. |

The built-in **Coated panel**, **Shifted laminate** and **Glass mosaic** are deterministic synthetic scenes. The first includes 16-bit analyzer images, an 8-bit coverage-bearing plane and an inclusion mask. The second includes known integer offsets. The third exercises complete 2 × 2 detector cells. These are demonstrations of calculations and input conditions, not physical measurements of real materials.

## Inputs and conventions

- Use unsigned grayscale linear PNG or single-page TIFF, or 8-bit equal-channel RGB. Inputs support 8- and 16-bit samples. Full alpha keeps a sample and incomplete alpha excludes it. Color photographs, gamma-encoded intensities, floating-point source TIFF, HDR stacks and automatic scanner profiles require external preparation.
- The detector canvas is the 0° image for plane imports and the cell grid for packed mosaics. Positions use zero-based detector pixels with X rightward and Y downward. Positive shift moves a source right or down. Raw maps show their original detector coordinates. Native crop copies the same detector window from all sources and does not preserve neighboring pixels outside the crop.
- Packed input uses complete sensor cells without interpolating angles between cells. Image size is limited to 5,000,000 pixels and 3200 pixels per side, with 14 MiB per encoded image. An imported source ZIP has a 95 MiB unpacking cap. A capture can hold 24 regions, 16 lines and 24 saved views.
- `S0` is the mean of the opposing pair sums. `Q=I0−I90`, `U=I45−I135`, and orientation is axial over 180°. Angle and degree use linear polarization only. Physical direction is determined by a known fixture and angle-order metadata. The strongest angle for a virtual analyzer is not the same as the maximum map calculated independently at every pixel.
- Registration handles bounded integer translations. The texture scorer proposes a candidate; inspector review is appropriate where textures repeat or captures have weak shared detail. Capture comparison assumes the two measurements already refer to the same detector coordinates.
- Rendered PNGs are display images. A linear TIFF preserves numerical values and uses NaN in unsupported positions. It is paired with a support mask in evidence ZIPs. A source package carries input images and current settings but omits saved inspection history.

## Structure

`polarbench/` implements Flask routes, acquisition codecs, detector preparation, Stokes reconstruction, quality and measurement tools, rendering and persistency. `static/` is the browser workspace. `scripts/count_backend.py` can be used to count the owned Python package independently of browser files and data.

The linear Stokes conventions are described in [Teledyne FLIR's polarization example](https://softwareservices.flir.com/Spinnaker/latest/examples/cpp/Polarization.html).
