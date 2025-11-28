# Carma: Fastest Lap Intelligence

Carma is a Streamlit dashboard we built to explore fastest laps at COTA and Barber. We took the Toyota GR dataset, added our own reference lines, generated fastest-lap parquet files, and turned everything into an interactive tool that lets you compare laps, replay “ghost” laps, and dig into driver style and sector deltas.

The goal was simple: make it easy (and fun) to understand where time is gained or lost on a lap.

---

## What the Dashboard Does

- Breaks down laps into micro-sectors and shows where drivers gain or lose time  
- Animates fastest laps with a ghost-car simulation  
- Lets you scrub through telemetry (speed, throttle, brake, gear)  
- Compares racing lines and inputs between drivers  
- Builds driver profiles with simple stats and radar charts  
- Supports both Barber and COTA with custom track maps we generated

---

## Data Used

We used the Toyota GR Hackathon dataset:

- **Endurance timing** (COTA + Barber): lap times, sectors, driver IDs  
- **Barber telemetry**: speed, throttle, brake, gear, GPS  
- **Our own additions**:
  - Barber centerline (built from GPS)
  - COTA reference polyline (built manually)
  - Fastest-lap parquet files we generated for speed/load efficiency

Because the raw telemetry files are huge (1–2 GB each), the telemetry file was filtered only to include the fastest laps necessary.

---

## How It Works (High-Level)

- Load endurance data → find fastest laps → compute sector and micro-sector deltas  
- Build track centerlines (our own for Barber + COTA)  
- Project telemetry onto a station-based coordinate system  
- Generate ghost-car animation frames from telemetry
- Create driver-profiling features from telemetry (throttle, brake, lateral G, steering, consistency)
- Plot everything in Streamlit with Plotly

If you’re just running it, you don’t need to worry about any of the internals.

---

## Running the App

Clone the repo:

```bash
git clone https://github.com/Ddhamani123/toyota-gr-carma.git
cd toyota-gr-carma
streamlit run toyota-gr-carma/toyota_hackathon_submission/toyota_hackathon_submission/Carma_Fastest_Lap_Dashboard.py

