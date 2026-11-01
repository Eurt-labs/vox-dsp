"""
vox-dsp Centralized Access Dashboard, Live Recording Studio & Voice Recognition Engine.
Zero-dependency HTTP server using Python's standard library.

Features:
- Minimalist Swiss/monochrome design with high data density
- Live in-browser microphone recording & instant DSP analysis
- Standalone downloadable plots for DFT, Spectrogram, Pitch, LPC, MFCCs, Vowel Space, and Glottal Source
- Raw DFT numerical data export (CSV)
- Complete one-click ZIP bundle export of all plots and reports
- Voice Recognition studio: Record pre-written sentences & authenticate/identify speakers
- Acoustic Info Guide: Complete plain-language explanations of all numbers and metrics
"""

import os
import sys
import json
import glob
import time
import urllib.parse
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
import soundfile as sf
import numpy as np

from analyze_voice import analyze_audio_signal, save_rows_to_csv, load_audio
from audio_dsp.diagnostic_engine import generate_voice_diagnostic_report
from dashboard import (
    render_voice_dashboard,
    render_dft_plot,
    render_spectrogram_plot,
    render_pitch_plot,
    render_lpc_plot,
    render_mfcc_plot,
    render_vowel_space_plot,
    render_glottal_plot,
    export_all_analyses_bundle,
)

from voice_recognition import (
    default_db,
    PHONETIC_SENTENCES,
    get_all_sentences,
)

PORT = 8050
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_available_audio_files():
    wavs = glob.glob(os.path.join(WORKSPACE_DIR, "*.wav"))
    wavs.sort(key=os.path.getmtime, reverse=True)
    return [os.path.basename(w) for w in wavs]


def load_dataset_records():
    csv_path = os.path.join(WORKSPACE_DIR, "voice_dataset.csv")
    if not os.path.isfile(csv_path):
        return []
    import csv
    with open(csv_path, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>vox-dsp | Voice Biometrics & Speaker Recognition</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    body { background-color: #09090b; color: #f4f4f5; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    .card-surface { background-color: #121215; border: 1px solid #27272a; }
    .nav-btn-active { border-bottom: 2px solid #f4f4f5; color: #f4f4f5; font-weight: 600; }
    .pulse-rec { animation: pulse 1.2s infinite ease-in-out; }
    @keyframes pulse { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.3; transform: scale(1.2); } }
  </style>
</head>
<body class="min-h-screen flex flex-col font-sans antialiased text-sm text-zinc-300">

  <!-- Minimal Header & Navigation -->
  <header class="border-b border-zinc-800 bg-zinc-950/80 backdrop-blur sticky top-0 z-50">
    <div class="max-w-6xl mx-auto px-4 h-14 flex items-center justify-between">
      <div class="flex items-center space-x-6">
        <div class="flex items-center space-x-2">
          <span class="font-mono text-xs tracking-widest uppercase text-zinc-400 font-bold">vox-dsp</span>
          <span class="text-zinc-600">/</span>
        </div>
        <nav class="flex space-x-5 text-xs font-mono uppercase tracking-wider">
          <button onclick="switchView('dashboard')" id="nav-dashboard" class="py-4 nav-btn-active">
            Acoustic Biometrics
          </button>
          <button onclick="switchView('recognition')" id="nav-recognition" class="py-4 text-zinc-500 hover:text-zinc-300">
            Voice Recognition
          </button>
          <button onclick="switchView('info')" id="nav-info" class="py-4 text-zinc-500 hover:text-zinc-300">
            Acoustic Info Guide
          </button>
        </nav>
      </div>

      <!-- Quick Actions for Active Dashboard -->
      <div id="headerQuickActions" class="flex items-center space-x-2">
        <select id="fileSelect" class="bg-zinc-900 border border-zinc-800 text-xs rounded px-2.5 py-1 text-zinc-300 font-mono">
          <option value="voice.wav">voice.wav</option>
        </select>
        <button onclick="toggleRecordDrawer()" class="border border-zinc-800 hover:border-zinc-700 bg-zinc-900 text-xs px-2.5 py-1 rounded text-zinc-300 flex items-center gap-1.5 font-medium">
          <span class="w-2 h-2 rounded-full bg-red-500"></span>
          <span>Record Voice</span>
        </button>
        <button id="btnAnalyze" onclick="triggerAnalyze()" class="bg-zinc-100 hover:bg-white text-zinc-950 font-semibold text-xs px-3 py-1 rounded transition">
          Analyze
        </button>
        <a id="btnDownloadZipHeader" href="/api/export/zip?file=voice.wav" class="border border-zinc-700 hover:border-zinc-500 bg-zinc-900 text-xs px-2.5 py-1 rounded text-zinc-200 transition font-mono flex items-center gap-1">
          <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path></svg>
          ZIP Bundle
        </a>
      </div>
    </div>
  </header>

  <!-- ============================================================== -->
  <!-- VIEW 1: ACOUSTIC BIOMETRICS DASHBOARD                          -->
  <!-- ============================================================== -->
  <div id="view-dashboard" class="flex-1 space-y-6">

    <!-- Live Recording Drawer -->
    <div id="recordingDrawer" class="hidden border-b border-zinc-800 bg-zinc-900/90 py-4 px-4 transition">
      <div class="max-w-6xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
        <div class="flex items-center space-x-3">
          <span class="w-2.5 h-2.5 rounded-full bg-red-500 pulse-rec"></span>
          <span id="recStatusLabel" class="text-xs uppercase font-mono text-red-400 font-semibold">Microphone Ready</span>
          <span id="recTimer" class="font-mono text-sm font-bold text-zinc-100 bg-zinc-950 px-2 py-0.5 rounded border border-zinc-800">00:00</span>
        </div>
        <canvas id="micCanvas" width="280" height="32" class="rounded bg-zinc-950 border border-zinc-800"></canvas>
        <div class="flex items-center space-x-2">
          <button id="btnStartRec" onclick="startMicRecording()" class="bg-red-500 hover:bg-red-600 text-white text-xs px-3 py-1 rounded font-medium">
            Start Recording
          </button>
          <button id="btnStopRec" onclick="stopMicRecording()" disabled class="bg-zinc-800 text-zinc-400 text-xs px-3 py-1 rounded font-medium opacity-50 cursor-not-allowed">
            Stop & Analyze
          </button>
          <button onclick="toggleRecordDrawer()" class="text-zinc-500 hover:text-zinc-300 text-xs px-2 py-1">
            Cancel
          </button>
        </div>
      </div>
    </div>

    <!-- Audio Playback Banner -->
    <div class="border-b border-zinc-800/50 bg-zinc-950/40 py-2.5 px-4">
      <div class="max-w-6xl mx-auto flex flex-wrap items-center justify-between gap-3 text-xs">
        <div class="flex items-center space-x-2 font-mono">
          <span class="text-zinc-500">SAMPLE:</span>
          <span id="activeFileLabel" class="text-zinc-200">voice.wav</span>
          <span id="activeDurationLabel" class="text-zinc-500">6.59s @ 16kHz</span>
        </div>
        <audio id="audioPlayer" controls class="h-7 w-72 rounded opacity-90"></audio>
        <div class="flex items-center space-x-2">
          <label class="border border-zinc-800 hover:border-zinc-700 text-zinc-400 px-2.5 py-1 rounded text-xs cursor-pointer font-mono">
            Upload WAV
            <input type="file" id="fileUploadInput" accept=".wav" class="hidden" onchange="handleFileUpload(event)">
          </label>
          <a id="btnZipDownloadBanner" href="/api/export/zip?file=voice.wav" class="border border-zinc-800 hover:border-zinc-700 text-zinc-300 px-2.5 py-1 rounded text-xs font-mono">
            Download All (ZIP)
          </a>
        </div>
      </div>
    </div>

    <!-- 4 Minimal Metric Cards -->
    <div class="max-w-6xl mx-auto px-4 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
      <div class="card-surface rounded-lg p-4">
        <div class="text-[11px] font-mono uppercase text-zinc-500">01 / Speaker Morphology</div>
        <div class="mt-2 text-base font-semibold text-zinc-100" id="cardSpeaker">Adult Male</div>
        <div class="mt-2 text-xs text-zinc-400 space-y-0.5">
          <div class="flex justify-between"><span>Vocal Tract Length:</span> <strong id="cardVTL" class="text-zinc-200 font-mono">17.2 cm</strong></div>
          <div class="flex justify-between"><span>Adapted LPC Order:</span> <strong id="cardLPCOrder" class="text-zinc-200 font-mono">18</strong></div>
        </div>
      </div>

      <div class="card-surface rounded-lg p-4">
        <div class="text-[11px] font-mono uppercase text-zinc-500">02 / Glottal Pitch (f0)</div>
        <div class="mt-2 text-base font-semibold text-zinc-100 flex items-baseline gap-1.5">
          <span id="cardF0" class="text-lg font-mono">117.1</span>
          <span class="text-xs text-zinc-500">Hz median</span>
        </div>
        <div class="mt-2 text-xs text-zinc-400 space-y-0.5">
          <div class="flex justify-between"><span>Prosodic Range:</span> <strong id="cardSpread" class="text-zinc-200 font-mono">11.45 st</strong></div>
          <div class="flex justify-between"><span>Voiced Activity:</span> <strong id="cardDuty" class="text-zinc-200 font-mono">23.0%</strong></div>
        </div>
      </div>

      <div class="card-surface rounded-lg p-4">
        <div class="text-[11px] font-mono uppercase text-zinc-500">03 / Formant Resonances</div>
        <div class="mt-2 text-xs font-mono text-zinc-200 font-semibold" id="cardFormants">F1: 435 | F2: 1595 Hz</div>
        <div class="mt-2 text-xs text-zinc-400 space-y-0.5">
          <div class="flex justify-between"><span>Dominant Vowel:</span> <strong id="cardVowel" class="text-zinc-200 truncate max-w-[120px] text-right">Mid-Central</strong></div>
          <div class="flex justify-between"><span>Vowel Space Area:</span> <strong id="cardVSA" class="text-zinc-200 font-mono">111,332 Hz²</strong></div>
        </div>
      </div>

      <div class="card-surface rounded-lg p-4">
        <div class="flex items-center justify-between text-[11px] font-mono uppercase text-zinc-500">
          <span>04 / Voice Stability</span>
          <span id="cardStatusBadge" class="text-[10px] font-mono px-1.5 py-0.5 rounded border border-zinc-700 text-zinc-300">BORDERLINE</span>
        </div>
        <div class="mt-2 text-base font-semibold text-zinc-100 flex items-baseline gap-1.5">
          <span id="cardJitter" class="text-lg font-mono">1.29%</span>
          <span class="text-xs text-zinc-500">jitter</span>
        </div>
        <div class="mt-2 text-xs text-zinc-400 space-y-0.5">
          <div class="flex justify-between"><span>Harmonicity (HNR):</span> <strong id="cardHNR" class="text-zinc-200 font-mono">10.99 dB</strong></div>
          <div class="flex justify-between"><span>Continuous CPPS:</span> <strong id="cardCPPS" class="text-zinc-200 font-mono">34.93 dB</strong></div>
        </div>
      </div>
    </div>

    <!-- Dashboard Sub-Tabs -->
    <div class="max-w-6xl mx-auto px-4 space-y-4">
      <div class="border-b border-zinc-800">
        <nav class="flex space-x-6 text-xs font-mono uppercase tracking-wider">
          <button onclick="switchDashTab('plots')" id="tabBtn-plots" class="py-2.5 nav-btn-active">
            DFT & Individual Analyses
          </button>
          <button onclick="switchDashTab('report')" id="tabBtn-report" class="py-2.5 text-zinc-500 hover:text-zinc-300">
            Clinical Evaluation
          </button>
          <button onclick="switchDashTab('visual')" id="tabBtn-visual" class="py-2.5 text-zinc-500 hover:text-zinc-300">
            6-Panel Composite
          </button>
          <button onclick="switchDashTab('dataset')" id="tabBtn-dataset" class="py-2.5 text-zinc-500 hover:text-zinc-300">
            Dataset Records (<span id="datasetCount">1</span>)
          </button>
        </nav>
      </div>

      <!-- SUB-TAB 1: DFT & INDIVIDUAL PLOTS GALLERY (NEW) -->
      <div id="tabContent-plots" class="space-y-4">
        
        <!-- Plot Selector Bar -->
        <div class="card-surface rounded-lg p-3 flex flex-wrap items-center justify-between gap-3">
          <div class="flex items-center space-x-2">
            <span class="text-xs font-mono uppercase text-zinc-400 font-semibold">Select Analysis:</span>
            <select id="analysisPlotSelect" onchange="updateSelectedPlot()" class="bg-zinc-950 border border-zinc-800 text-xs rounded px-2.5 py-1 text-zinc-200 font-mono">
              <option value="dft">1. Discrete Fourier Transform (DFT) Magnitude Spectrum</option>
              <option value="spectrogram">2. STFT Time-Frequency Spectrogram (0-4000 Hz)</option>
              <option value="pitch">3. Fundamental Pitch (f0) Intonation Contour</option>
              <option value="lpc">4. LPC Vocal Tract Filter Envelope vs FFT Harmonics</option>
              <option value="mfcc">5. Normalized MFCCs (c1-c12) & Log Energy Contour</option>
              <option value="vowel_space">6. Phonetic IPA Vowel Space Quadrilateral</option>
              <option value="glottal">7. Glottal Flow Excitation Residual (Inverse Filter)</option>
            </select>
          </div>

          <div class="flex items-center space-x-2">
            <a id="btnDownloadCurrentPlot" href="/api/plot/dft?file=voice.wav&download=1" download class="border border-zinc-700 hover:border-zinc-500 bg-zinc-900 text-xs px-3 py-1 rounded text-zinc-200 font-mono flex items-center gap-1.5 transition">
              <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path></svg>
              Download Plot (PNG)
            </a>
            <a id="btnDownloadDftCsv" href="/api/data/dft_csv?file=voice.wav" download class="border border-zinc-700 hover:border-zinc-500 bg-zinc-900 text-xs px-3 py-1 rounded text-zinc-200 font-mono flex items-center gap-1.5 transition">
              <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg>
              Download Raw DFT Data (CSV)
            </a>
          </div>
        </div>

        <!-- Plot Display Canvas -->
        <div class="card-surface rounded-lg p-4 flex flex-col items-center space-y-3">
          <div class="w-full flex justify-between items-center text-xs font-mono text-zinc-400">
            <span id="plotTitleLabel">Discrete Fourier Transform (DFT) Magnitude Spectrum</span>
            <span id="plotSubLabel" class="text-zinc-500">Target Vowel Slice | N=512 (32ms) Hann Window | fs=16kHz</span>
          </div>
          <div class="w-full border border-zinc-800 rounded bg-zinc-950 p-1 flex justify-center min-h-[300px] items-center">
            <img id="activeAnalysisPlotImg" src="/api/plot/dft?file=voice.wav" alt="Analysis Plot" class="max-w-full h-auto rounded transition">
          </div>
          <p id="plotDescriptionText" class="text-xs text-zinc-400 w-full leading-relaxed border-t border-zinc-800/80 pt-2">
            The DFT spectrum converts the stationary 32 ms windowed vowel frame into positive frequency bins. The prominent vertical red line marks the dominant glottal pitch harmonic, and gray dotted lines indicate upper integer multiples (2f0, 3f0, etc.). Raw frequency bins and decibel levels can be exported via the CSV button above.
          </p>
        </div>

      </div>

      <!-- SUB-TAB 2: CLINICAL REPORT -->
      <div id="tabContent-report" class="hidden space-y-4">
        <div class="card-surface rounded-lg p-5 space-y-4">
          <div class="flex items-center justify-between border-b border-zinc-800 pb-3">
            <span class="text-xs font-mono uppercase text-zinc-400 font-semibold">ASHA & Praat Normative Evaluation</span>
            <span id="evalOverallHealth" class="text-xs text-zinc-400 font-mono"></span>
          </div>
          <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
            <div class="space-y-2 border border-zinc-800 rounded p-3 bg-zinc-950/40">
              <div class="text-[11px] font-mono uppercase text-zinc-500 font-semibold">Vocal Stability Criteria</div>
              <div class="flex justify-between py-1 border-b border-zinc-900"><span>Jitter (local, %):</span><span id="repJitter" class="font-mono text-zinc-200">1.29%</span></div>
              <div class="flex justify-between py-1 border-b border-zinc-900"><span>Shimmer (local, %):</span><span id="repShimmer" class="font-mono text-zinc-200">14.75%</span></div>
              <div class="flex justify-between py-1 border-b border-zinc-900"><span>Harmonics-to-Noise Ratio:</span><span id="repHNR" class="font-mono text-zinc-200">10.99 dB</span></div>
              <div class="flex justify-between py-1"><span>Continuous CPPS:</span><span id="repCPPS" class="font-mono text-zinc-200">34.93 dB</span></div>
            </div>
            <div class="space-y-2 border border-zinc-800 rounded p-3 bg-zinc-950/40">
              <div class="text-[11px] font-mono uppercase text-zinc-500 font-semibold">Resonance & Glottal Source</div>
              <div class="flex justify-between py-1 border-b border-zinc-900"><span>Glottal Flow NAQ:</span><span id="repNAQ" class="font-mono text-zinc-200">0.092</span></div>
              <div class="flex justify-between py-1 border-b border-zinc-900"><span>Vocal Tract Length:</span><span id="repVTL" class="font-mono text-zinc-200">17.2 cm</span></div>
              <div class="flex justify-between py-1 border-b border-zinc-900"><span>Prosodic Style:</span><span id="repProsody" class="font-mono text-zinc-200">Dynamic</span></div>
              <div class="flex justify-between py-1"><span>Phonetic VSA:</span><span id="repVSA" class="font-mono text-zinc-200">111,332 Hz²</span></div>
            </div>
          </div>
          <div>
            <div class="flex items-center justify-between mb-2">
              <span class="text-[11px] font-mono uppercase text-zinc-500">Diagnostic Narrative</span>
              <button onclick="copyReportText()" class="text-xs text-zinc-400 hover:text-zinc-200 underline font-mono">Copy Report</button>
            </div>
            <pre id="rawReportText" class="p-3.5 rounded bg-zinc-950 border border-zinc-800 font-mono text-xs text-zinc-300 leading-relaxed overflow-x-auto"></pre>
          </div>
        </div>
      </div>

      <!-- SUB-TAB 3: 6-PANEL COMPOSITE -->
      <div id="tabContent-visual" class="hidden space-y-3">
        <div class="card-surface rounded-lg p-4 flex flex-col items-center">
          <div class="w-full flex justify-between items-center mb-3">
            <span class="text-xs font-mono uppercase text-zinc-400 font-semibold">6-Panel Composite Diagnostic Dashboard</span>
            <a id="btnDownloadComposite" href="/api/plot/composite?file=voice.wav&download=1" download class="border border-zinc-800 bg-zinc-900 text-xs px-3 py-1 rounded text-zinc-200 font-mono">Download PNG</a>
          </div>
          <div class="w-full border border-zinc-800 rounded bg-zinc-950 p-1 flex justify-center">
            <img id="dashboardImg" src="/api/plot/composite?file=voice.wav" alt="6-Panel Diagnostic Dashboard" class="max-w-full h-auto rounded">
          </div>
        </div>
      </div>

      <!-- SUB-TAB 4: DATASET ARCHIVE -->
      <div id="tabContent-dataset" class="hidden space-y-3">
        <div class="card-surface rounded-lg p-4 space-y-3">
          <div class="flex justify-between items-center">
            <span class="text-xs font-mono uppercase text-zinc-400 font-semibold">voice_dataset.csv Archive</span>
            <a href="/voice_dataset.csv" download class="border border-zinc-800 text-xs px-2.5 py-1 rounded font-mono text-zinc-300">Download CSV</a>
          </div>
          <div class="overflow-x-auto border border-zinc-800 rounded">
            <table class="min-w-full text-xs text-left text-zinc-300 font-mono">
              <thead class="bg-zinc-950 text-zinc-500 uppercase border-b border-zinc-800">
                <tr>
                  <th class="px-3 py-2">File</th>
                  <th class="px-3 py-2">Speaker</th>
                  <th class="px-3 py-2">VTL</th>
                  <th class="px-3 py-2">f0 (Hz)</th>
                  <th class="px-3 py-2">F1</th>
                  <th class="px-3 py-2">F2</th>
                  <th class="px-3 py-2">Jitter%</th>
                  <th class="px-3 py-2">HNR</th>
                  <th class="px-3 py-2">CPPS</th>
                  <th class="px-3 py-2">Assessment</th>
                </tr>
              </thead>
              <tbody id="datasetTableBody" class="divide-y divide-zinc-900"></tbody>
            </table>
          </div>
        </div>
      </div>

    </div>
  </div>

  <!-- ============================================================== -->
  <!-- VIEW 2: VOICE RECOGNITION & SPEAKER IDENTIFICATION             -->
  <!-- ============================================================== -->
  <div id="view-recognition" class="hidden flex-1 max-w-6xl w-full mx-auto px-4 py-6 space-y-6">

    <div class="card-surface rounded-xl p-5 space-y-4">
      <div class="flex items-center justify-between border-b border-zinc-800/80 pb-3">
        <div class="flex items-center space-x-2">
          <span class="font-mono text-xs uppercase tracking-wider text-zinc-400 font-semibold">Pre-Written Benchmark Sentence</span>
          <span id="promptCategory" class="text-[10px] font-mono px-2 py-0.5 rounded bg-zinc-900 border border-zinc-800 text-zinc-400">All-Phoneme Pangram</span>
        </div>
        <button onclick="cycleSentence()" class="text-xs font-mono text-zinc-400 hover:text-zinc-100 flex items-center gap-1">
          <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path></svg>
          Next Sentence
        </button>
      </div>

      <div class="p-4 rounded-lg bg-zinc-950 border border-zinc-800 text-center">
        <p class="text-xs text-zinc-500 uppercase tracking-widest font-mono mb-1">Please read aloud:</p>
        <p id="promptText" class="text-lg font-medium text-zinc-100 tracking-wide font-sans">"The quick brown fox jumps over the lazy dog."</p>
        <p id="promptFocus" class="text-[11px] text-zinc-500 font-mono mt-2">Phonetic Focus: Full consonant and vowel articulatory distribution</p>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <!-- Recognize Voice -->
        <div class="border border-zinc-800 rounded-lg p-4 bg-zinc-950/40 space-y-3 flex flex-col justify-between">
          <div>
            <div class="text-xs font-mono uppercase tracking-wider text-zinc-300 font-semibold mb-1">Mode 1: Recognize Voice</div>
            <p class="text-xs text-zinc-400">Speak the displayed sentence to match your voice against all enrolled speaker fingerprints.</p>
          </div>
          <div class="space-y-2 pt-2">
            <button id="btnRecogStart" onclick="startRecognitionRecord()" class="w-full bg-zinc-100 hover:bg-white text-zinc-950 font-semibold text-xs py-2 rounded transition flex items-center justify-center gap-2">
              <span class="w-2 h-2 rounded-full bg-red-500"></span>
              <span id="btnRecogLabel">Record & Identify Voice</span>
            </button>
            <div id="recogTimer" class="hidden text-center font-mono text-xs text-red-400 font-bold">REC 00:00</div>
          </div>
        </div>

        <!-- Enroll Speaker -->
        <div class="border border-zinc-800 rounded-lg p-4 bg-zinc-950/40 space-y-3">
          <div>
            <div class="text-xs font-mono uppercase tracking-wider text-zinc-300 font-semibold mb-1">Mode 2: Enroll New Speaker</div>
            <p class="text-xs text-zinc-400">Register a new voice profile into the biometric database.</p>
          </div>
          <div class="space-y-2">
            <input type="text" id="speakerNameInput" placeholder="Speaker Name (e.g. Dhruv)" class="w-full bg-zinc-900 border border-zinc-800 rounded px-3 py-1.5 text-xs text-zinc-200 focus:outline-none focus:border-zinc-600 font-mono">
            <button id="btnEnrollStart" onclick="startEnrollRecord()" class="w-full border border-zinc-700 hover:border-zinc-500 bg-zinc-900 text-zinc-200 font-medium text-xs py-2 rounded transition flex items-center justify-center gap-2">
              <span class="w-2 h-2 rounded-full bg-red-500"></span>
              <span id="btnEnrollLabel">Record & Enroll Voice</span>
            </button>
            <div id="enrollTimer" class="hidden text-center font-mono text-xs text-red-400 font-bold">REC 00:00</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Recognition Result Display Card -->
    <div id="recogResultCard" class="hidden card-surface rounded-xl p-5 space-y-4">
      <div class="flex items-center justify-between border-b border-zinc-800 pb-3">
        <span class="text-xs font-mono uppercase tracking-wider text-zinc-400 font-semibold">Biometric Identification Result</span>
        <span id="recogMatchBadge" class="text-xs font-mono px-2 py-0.5 rounded border border-zinc-700 text-zinc-200 font-bold">AUTHENTICATED</span>
      </div>

      <div class="p-4 rounded-lg bg-zinc-950 border border-zinc-800 flex flex-col sm:flex-row items-center justify-between gap-4">
        <div>
          <div class="text-xs text-zinc-500 uppercase font-mono">Identified Speaker</div>
          <div id="recogSpeakerName" class="text-2xl font-bold text-zinc-100 font-mono mt-1">Dhruv</div>
          <div id="recogMessage" class="text-xs text-zinc-400 mt-1">Matched with 98.4% biometric confidence.</div>
        </div>
        <div class="text-right">
          <div class="text-xs text-zinc-500 uppercase font-mono">Confidence Score</div>
          <div id="recogScore" class="text-3xl font-mono font-bold text-zinc-100">98.4%</div>
        </div>
      </div>

      <div>
        <div class="text-xs font-mono uppercase tracking-wider text-zinc-500 mb-2">Candidate Comparison Rankings</div>
        <div class="overflow-x-auto border border-zinc-800 rounded">
          <table class="min-w-full text-xs font-mono text-left text-zinc-300">
            <thead class="bg-zinc-950 text-zinc-500 uppercase border-b border-zinc-800">
              <tr>
                <th class="px-3 py-2">Candidate Name</th>
                <th class="px-3 py-2">Cosine Similarity</th>
                <th class="px-3 py-2">Match Confidence</th>
                <th class="px-3 py-2">Enrolled Pitch</th>
                <th class="px-3 py-2">Status</th>
              </tr>
            </thead>
            <tbody id="recogCandidatesBody" class="divide-y divide-zinc-900"></tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- Enrolled Speakers Table -->
    <div class="card-surface rounded-xl p-5 space-y-3">
      <div class="flex items-center justify-between">
        <span class="text-xs font-mono uppercase tracking-wider text-zinc-400 font-semibold">Enrolled Biometric Profiles (<span id="enrolledCount">0</span>)</span>
      </div>
      <div class="overflow-x-auto border border-zinc-800 rounded">
        <table class="min-w-full text-xs font-mono text-left text-zinc-300">
          <thead class="bg-zinc-950 text-zinc-500 uppercase border-b border-zinc-800">
            <tr>
              <th class="px-3 py-2">Speaker Name</th>
              <th class="px-3 py-2">Enrolled Date</th>
              <th class="px-3 py-2">Pitch (f0)</th>
              <th class="px-3 py-2">Sample Clip</th>
              <th class="px-3 py-2 text-right">Action</th>
            </tr>
          </thead>
          <tbody id="enrolledSpeakersBody" class="divide-y divide-zinc-900"></tbody>
        </table>
      </div>
    </div>

  </div>

  <!-- ============================================================== -->
  <!-- VIEW 3: ACOUSTIC INFO GUIDE                                    -->
  <!-- ============================================================== -->
  <div id="view-info" class="hidden flex-1 max-w-6xl w-full mx-auto px-4 py-6 space-y-6">
    <div class="card-surface rounded-xl p-6 space-y-6">
      <div class="border-b border-zinc-800 pb-3">
        <h2 class="text-base font-bold text-zinc-100 uppercase tracking-wider font-mono">Acoustic Metrics & Biometrics Reference Guide</h2>
        <p class="text-xs text-zinc-400 mt-1">Detailed physical and clinical definitions of every metric, ratio, and threshold extracted by vox-dsp.</p>
      </div>

      <div class="space-y-3">
        <div class="text-xs font-mono uppercase tracking-widest text-zinc-400 font-bold border-l-2 border-zinc-400 pl-2">
          01 / Discrete Fourier Transform (DFT) & Spectral Decomposition
        </div>
        <div class="p-3.5 bg-zinc-950 border border-zinc-800 rounded text-xs leading-relaxed text-zinc-400 space-y-1.5">
          <p><strong class="text-zinc-200">Discrete Fourier Transform (DFT):</strong> Converts a finite time-domain speech frame (N = 512 samples, 32.0 ms) into N/2 + 1 = 257 discrete positive frequency bins spaced at Δf = fs/N = 31.25 Hz.</p>
          <p><strong class="text-zinc-200">Hann Windowing:</strong> Tapers the frame boundaries to zero using w[n] = 0.5 - 0.5 * cos(2πn / (N-1)), preventing artificial high-frequency spectral edge leakage.</p>
          <p><strong class="text-zinc-200">Dominant Peak & Harmonics:</strong> In active voiced vowels, the dominant peak corresponds to either the fundamental glottal pulse frequency ($f_0$) or the first vocal tract formant resonance ($F_1$). Successive peaks appear at integer multiples ($2f_0, 3f_0, 4f_0$), forming the acoustic harmonic series.</p>
        </div>
      </div>

      <div class="space-y-3">
        <div class="text-xs font-mono uppercase tracking-widest text-zinc-400 font-bold border-l-2 border-zinc-400 pl-2">
          02 / Glottal Fundamental Frequency ($f_0$) & Prosody
        </div>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs leading-relaxed text-zinc-400">
          <div class="p-3.5 bg-zinc-950 border border-zinc-800 rounded space-y-1">
            <strong class="text-zinc-200 block text-xs">Fundamental Frequency ($f_0$):</strong>
            The rate at which the vocal folds open and close per second (measured in Hertz).
            <ul class="list-disc list-inside mt-1 text-zinc-400 space-y-0.5">
              <li><strong>Adult Male:</strong> 85 – 155 Hz (Sample speaker: ~117.1 Hz).</li>
              <li><strong>Adult Female:</strong> 165 – 255 Hz.</li>
              <li><strong>Child:</strong> 250 – 400+ Hz.</li>
            </ul>
          </div>
          <div class="p-3.5 bg-zinc-950 border border-zinc-800 rounded space-y-1">
            <strong class="text-zinc-200 block text-xs">Prosodic Pitch Range (Semitones):</strong>
            Logarithmic pitch dynamic width: 12 * log2(f0_max / f0_min).
            <ul class="list-disc list-inside mt-1 text-zinc-400 space-y-0.5">
              <li><strong>&lt; 3.0 semitones:</strong> Monotone, flat, or depressed speech delivery.</li>
              <li><strong>3.0 – 9.0 semitones:</strong> Typical conversational intonation.</li>
              <li><strong>&gt; 9.0 semitones:</strong> Highly expressive, emotive, or animated delivery.</li>
            </ul>
          </div>
        </div>
      </div>

      <div class="space-y-3">
        <div class="text-xs font-mono uppercase tracking-widest text-zinc-400 font-bold border-l-2 border-zinc-400 pl-2">
          03 / Vocal Tract Formants ($F_1, F_2, F_3$) & VSA
        </div>
        <div class="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs leading-relaxed text-zinc-400">
          <div class="p-3.5 bg-zinc-950 border border-zinc-800 rounded space-y-1">
            <strong class="text-zinc-200 block text-xs">First Formant ($F_1$):</strong>
            Controlled by <strong>jaw opening</strong> and pharynx size:
            <ul class="list-disc list-inside mt-1 space-y-0.5">
              <li><strong>High $F_1$ (600–800 Hz):</strong> Open jaw (/ɑ/, /æ/, /ʌ/).</li>
              <li><strong>Low $F_1$ (250–350 Hz):</strong> Closed jaw (/i/, /u/).</li>
            </ul>
          </div>
          <div class="p-3.5 bg-zinc-950 border border-zinc-800 rounded space-y-1">
            <strong class="text-zinc-200 block text-xs">Second Formant ($F_2$):</strong>
            Controlled by <strong>tongue position (front vs. back)</strong>:
            <ul class="list-disc list-inside mt-1 space-y-0.5">
              <li><strong>High $F_2$ (1800–2400 Hz):</strong> Front tongue (/i/, /e/).</li>
              <li><strong>Low $F_2$ (800–1200 Hz):</strong> Back tongue (/u/, /o/, /ɑ/).</li>
            </ul>
          </div>
          <div class="p-3.5 bg-zinc-950 border border-zinc-800 rounded space-y-1">
            <strong class="text-zinc-200 block text-xs">Vowel Space Area (VSA):</strong>
            The geometric area ($\text{Hz}^2$) bounded by corner vowel formants. Larger VSA indicates clear articulatory precision; compressed VSA is a biomarker for speech motor fatigue or dysarthria.
          </div>
        </div>
      </div>

      <div class="space-y-3">
        <div class="text-xs font-mono uppercase tracking-widest text-zinc-400 font-bold border-l-2 border-zinc-400 pl-2">
          04 / Clinical Perturbation & Dysphonia (ASHA Criteria)
        </div>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs leading-relaxed text-zinc-400">
          <div class="p-3.5 bg-zinc-950 border border-zinc-800 rounded space-y-1">
            <strong class="text-zinc-200 block text-xs">Jitter (local, %): Cycle-to-Cycle Pitch Perturbation</strong>
            Microscopic variations in consecutive vocal fold vibration periods.
            <ul class="list-disc list-inside mt-1 space-y-0.5">
              <li><strong>&lt; 1.04%:</strong> Normal sustained modal phonation.</li>
              <li><strong>&gt; 1.50%:</strong> Elevated frequency perturbation (vocal roughness, fatigue).</li>
            </ul>
          </div>
          <div class="p-3.5 bg-zinc-950 border border-zinc-800 rounded space-y-1">
            <strong class="text-zinc-200 block text-xs">Shimmer (local, %): Cycle-to-Cycle Amplitude Perturbation</strong>
            Microscopic fluctuations in peak acoustic pressure from cycle to cycle.
            <ul class="list-disc list-inside mt-1 space-y-0.5">
              <li><strong>&lt; 3.81%:</strong> Normal sustained phonation.</li>
              <li><strong>&gt; 5.50%:</strong> Elevated amplitude variation (incomplete glottal closure).</li>
            </ul>
          </div>
          <div class="p-3.5 bg-zinc-950 border border-zinc-800 rounded space-y-1">
            <strong class="text-zinc-200 block text-xs">Harmonics-to-Noise Ratio (HNR, dB):</strong>
            Ratio of periodic harmonic energy produced by vocal folds to turbulent aspiration noise:
            <ul class="list-disc list-inside mt-1 space-y-0.5">
              <li><strong>&gt; 15.0 – 20.0 dB:</strong> Clean harmonic purity with negligible air turbulence.</li>
              <li><strong>&lt; 11.0 dB:</strong> Significant turbulent aspiration noise or hoarseness.</li>
            </ul>
          </div>
          <div class="p-3.5 bg-zinc-950 border border-zinc-800 rounded space-y-1">
            <strong class="text-zinc-200 block text-xs">Continuous CPPS (Cepstral Peak Prominence, dB):</strong>
            The ASHA gold standard for dysphonia in continuous conversational speech. Values <strong>&gt; 10.0 dB</strong> confirm clear periodic vocal fold vibration.
          </div>
        </div>
      </div>
    </div>
  </div>

  <footer class="border-t border-zinc-800/60 py-3 text-center text-xs font-mono text-zinc-600">
    vox-dsp / minimal acoustic biometrics & speaker recognition
  </footer>

  <script>
    let currentProfile = null;
    let sentencesList = [];
    let currentSentenceIndex = 0;

    let mediaStream = null;
    let audioContext = null;
    let mediaRecorder = null;
    let recordedChunks = [];
    let recordInterval = null;
    let recordSeconds = 0;
    let animFrame = null;
    let micAnalyser = null;

    const plotDescriptions = {
      dft: {
        title: "Discrete Fourier Transform (DFT) Magnitude Spectrum",
        sub: "Target Vowel Slice | N=512 (32ms) Hann Window | fs=16kHz",
        desc: "The DFT spectrum decomposes a stationary 32 ms windowed vowel frame into positive frequency bins. The vertical dashed red line marks the dominant pitch harmonic, and dotted gray lines indicate integer harmonics. Download the raw CSV data to analyze exact frequency and decibel points."
      },
      spectrogram: {
        title: "Short-Time Fourier Transform (STFT) Speech Spectrogram",
        sub: "0 - 4000 Hz Speech Band | 75% Overlap | Magma Heatmap",
        desc: "Displays the continuous spectral energy distribution across time and frequency. Bright horizontal bands represent vocal tract formants during voiced vowels, while diffuse high-frequency energy represents unvoiced fricatives and stops."
      },
      pitch: {
        title: "Glottal Fundamental Frequency (f0) Intonation Contour",
        sub: "Normalized Autocorrelation & Parabolic Sub-sample Interpolation",
        desc: "Traces the speaker's vocal pitch intonation over time. The solid blue line marks the median fundamental pitch (f0), and the shaded blue band represents the ±1 standard deviation prosodic variance."
      },
      lpc: {
        title: "Linear Predictive Coding (LPC) Vocal Tract Filter Envelope",
        sub: "Pre-Emphasized Levinson-Durbin Toeplitz Solver (p=18)",
        desc: "Deconvolves the vocal tract filter from the glottal source. The bold blue curve fits a smooth spectral envelope over the gray FFT harmonics, locating resonant formant frequencies (F1 jaw opening, F2 tongue position, F3 lip/pharynx)."
      },
      mfcc: {
        title: "Normalized Mel-Frequency Cepstral Coefficients (c1 - c12, CMVN)",
        sub: "Log Mel Filterbank Liftering & Cepstral Normalization",
        desc: "Maps spectral timbre onto the human perceptual Mel scale. Cepstral Mean & Variance Normalization (CMVN) isolates phonetic articulation from frame loudness (c0, shown in the lower panel)."
      },
      vowel_space: {
        title: "Phonetic Vowel Space Quadrilateral (F1 vs F2 IPA Acoustic Space)",
        sub: "Inverted Phonetic F1/F2 Space with Cardinal Vowel Benchmarks",
        desc: "Plots the speaker's formant dispersion against standard International Phonetic Alphabet (IPA) cardinal vowels (/i/, /u/, /æ/, /ɑ/). The green star marks the speaker's acoustic centroid and the bounded area represents the Vowel Space Area (VSA)."
      },
      glottal: {
        title: "Glottal Flow Derivative Residual Waveform (Inverse Filtered Source)",
        sub: "Vocal Fold Excitation Source Signal e[n] after LPC Deconvolution",
        desc: "Extracts the raw acoustic pulse train generated at the vocal folds before filtering by the throat and mouth. The negative derivative spikes correspond to glottal closure instants (GCI), with Normalized Amplitude Quotient (NAQ) quantifying closure velocity."
      }
    };

    async function initApp() {
      await refreshFileList();
      await loadSentences();
      await loadEnrolledSpeakers();
      updateActiveAudio();
      triggerAnalyze();
      loadDatasetTable();
      updateSelectedPlot();
    }

    function switchView(viewId) {
      ['dashboard', 'recognition', 'info'].forEach(v => {
        document.getElementById(`view-${v}`).classList.add('hidden');
        document.getElementById(`nav-${v}`).className = 'py-4 text-zinc-500 hover:text-zinc-300';
      });
      document.getElementById(`view-${viewId}`).classList.remove('hidden');
      document.getElementById(`nav-${viewId}`).className = 'py-4 nav-btn-active';

      const actions = document.getElementById('headerQuickActions');
      if (viewId === 'dashboard') actions.classList.remove('hidden');
      else actions.classList.add('hidden');
    }

    function switchDashTab(tabId) {
      ['plots', 'report', 'visual', 'dataset'].forEach(t => {
        document.getElementById(`tabContent-${t}`).classList.add('hidden');
        document.getElementById(`tabBtn-${t}`).className = 'py-2.5 text-zinc-500 hover:text-zinc-300';
      });
      document.getElementById(`tabContent-${tabId}`).classList.remove('hidden');
      document.getElementById(`tabBtn-${tabId}`).className = 'py-2.5 nav-btn-active';
    }

    async function refreshFileList() {
      try {
        const res = await fetch('/api/files');
        const files = await res.json();
        const sel = document.getElementById('fileSelect');
        const prev = sel.value;
        sel.innerHTML = '';
        files.forEach(f => {
          const opt = document.createElement('option');
          opt.value = f;
          opt.textContent = f;
          sel.appendChild(opt);
        });
        if (files.includes(prev)) sel.value = prev;
      } catch (e) {
        console.error(e);
      }
    }

    function updateActiveAudio() {
      const file = document.getElementById('fileSelect').value || 'voice.wav';
      document.getElementById('activeFileLabel').textContent = file;
      document.getElementById('audioPlayer').src = `/api/audio?file=${encodeURIComponent(file)}`;

      // Update download links
      document.getElementById('btnDownloadZipHeader').href = `/api/export/zip?file=${encodeURIComponent(file)}`;
      document.getElementById('btnZipDownloadBanner').href = `/api/export/zip?file=${encodeURIComponent(file)}`;
      document.getElementById('btnDownloadDftCsv').href = `/api/data/dft_csv?file=${encodeURIComponent(file)}`;
      document.getElementById('btnDownloadComposite').href = `/api/plot/composite?file=${encodeURIComponent(file)}&download=1`;
      document.getElementById('dashboardImg').src = `/api/plot/composite?file=${encodeURIComponent(file)}&t=${Date.now()}`;

      updateSelectedPlot();
    }

    function updateSelectedPlot() {
      const file = document.getElementById('fileSelect').value || 'voice.wav';
      const plotType = document.getElementById('analysisPlotSelect').value || 'dft';

      const img = document.getElementById('activeAnalysisPlotImg');
      img.style.opacity = '0.4';
      img.src = `/api/plot/${plotType}?file=${encodeURIComponent(file)}&t=${Date.now()}`;
      img.onload = () => { img.style.opacity = '1.0'; };

      document.getElementById('btnDownloadCurrentPlot').href = `/api/plot/${plotType}?file=${encodeURIComponent(file)}&download=1`;

      const csvBtn = document.getElementById('btnDownloadDftCsv');
      if (plotType === 'dft') {
        csvBtn.classList.remove('hidden');
        csvBtn.href = `/api/data/dft_csv?file=${encodeURIComponent(file)}`;
      } else {
        csvBtn.classList.add('hidden');
      }

      const meta = plotDescriptions[plotType] || plotDescriptions['dft'];
      document.getElementById('plotTitleLabel').textContent = meta.title;
      document.getElementById('plotSubLabel').textContent = meta.sub;
      document.getElementById('plotDescriptionText').textContent = meta.desc;
    }

    document.getElementById('fileSelect').addEventListener('change', () => {
      updateActiveAudio();
      triggerAnalyze();
    });

    async function triggerAnalyze() {
      const file = document.getElementById('fileSelect').value || 'voice.wav';
      const btn = document.getElementById('btnAnalyze');
      btn.textContent = '...';
      btn.disabled = true;

      try {
        const res = await fetch(`/api/analyze?file=${encodeURIComponent(file)}`);
        const data = await res.json();
        currentProfile = data;
        populateDashboard(data);
      } catch (err) {
        alert('Analysis error: ' + err);
      } finally {
        btn.textContent = 'Analyze';
        btn.disabled = false;
      }
    }

    function populateDashboard(data) {
      const demo = data.demographics || {};
      const pitch = data.pitch || {};
      const formants = data.formants || {};
      const clinical = data.clinical_vowel_biometrics || {};
      const cont = data.continuous_speech_biometrics || {};
      const diag = data.diagnostic_evaluation || {};

      document.getElementById('activeDurationLabel').textContent = `${data.duration_s || 0}s @ ${data.sampling_rate || 16000}Hz`;

      document.getElementById('cardSpeaker').textContent = demo.demographic_class || 'Adult Male';
      document.getElementById('cardVTL').textContent = `${demo.vocal_tract_length_cm || 17.2} cm`;
      document.getElementById('cardLPCOrder').textContent = demo.adapted_lpc_order || 18;

      document.getElementById('cardF0').textContent = pitch.median_f0 || 0;
      document.getElementById('cardSpread').textContent = `${pitch.pitch_range_semitones || 0} st`;
      document.getElementById('cardDuty').textContent = `${((pitch.voiced_ratio || 0) * 100).toFixed(1)}%`;

      document.getElementById('cardFormants').textContent = `F1: ${Math.round(formants.f1_hz || 0)} | F2: ${Math.round(formants.f2_hz || 0)} Hz`;
      document.getElementById('cardVowel').textContent = formants.dominant_vowel || 'N/A';
      document.getElementById('cardVSA').textContent = `${(formants.vowel_space_area_hz2 || 0).toLocaleString()} Hz²`;

      document.getElementById('cardJitter').textContent = `${clinical.jitter_local_percent || 0}%`;
      document.getElementById('cardHNR').textContent = `${clinical.hnr_db || 0} dB`;
      document.getElementById('cardCPPS').textContent = `${cont.cpps_db || 0} dB`;
      document.getElementById('cardStatusBadge').textContent = (diag.jitter_flag || 'NORMAL').toUpperCase();

      document.getElementById('evalOverallHealth').textContent = diag.overall_health || '';
      document.getElementById('repJitter').textContent = `${clinical.jitter_local_percent}% [${diag.jitter_flag || 'Pass'}]`;
      document.getElementById('repShimmer').textContent = `${clinical.shimmer_local_percent}% [${diag.shimmer_flag || 'Pass'}]`;
      document.getElementById('repHNR').textContent = `${clinical.hnr_db} dB [${diag.hnr_flag || 'Pass'}]`;
      document.getElementById('repCPPS').textContent = `${cont.cpps_db} dB [${diag.cpps_flag || 'Pass'}]`;

      document.getElementById('repNAQ').textContent = `${cont.normalized_amplitude_quotient || 0.09} (Glottal Closure)`;
      document.getElementById('repVTL').textContent = `${demo.vocal_tract_length_cm} cm (${demo.demographic_class})`;
      document.getElementById('repProsody').textContent = diag.intonation_style || 'Normal';
      document.getElementById('repVSA').textContent = `${(formants.vowel_space_area_hz2 || 0).toLocaleString()} Hz²`;

      document.getElementById('rawReportText').textContent = data.report_text || '';
    }

    async function loadDatasetTable() {
      try {
        const res = await fetch('/api/dataset');
        const rows = await res.json();
        document.getElementById('datasetCount').textContent = rows.length;
        const tbody = document.getElementById('datasetTableBody');
        tbody.innerHTML = '';
        rows.forEach(r => {
          const tr = document.createElement('tr');
          tr.className = 'hover:bg-zinc-900/60 transition';
          tr.innerHTML = `
            <td class="px-3 py-1.5 text-zinc-100">${r.File || ''}</td>
            <td class="px-3 py-1.5">${r.Demographic || ''}</td>
            <td class="px-3 py-1.5">${r.Est_VTL_cm || ''}</td>
            <td class="px-3 py-1.5">${r.Pitch_Median_Hz || ''}</td>
            <td class="px-3 py-1.5">${r.Formant_F1_Hz || ''}</td>
            <td class="px-3 py-1.5">${r.Formant_F2_Hz || ''}</td>
            <td class="px-3 py-1.5">${r.Jitter_Local_Pct || ''}</td>
            <td class="px-3 py-1.5">${r.HNR_dB || ''}</td>
            <td class="px-3 py-1.5">${r.CPPS_dB || ''}</td>
            <td class="px-3 py-1.5 truncate max-w-xs text-zinc-500">${r.Clinical_Assessment || ''}</td>
          `;
          tbody.appendChild(tr);
        });
      } catch (e) {
        console.error(e);
      }
    }

    function copyReportText() {
      const text = document.getElementById('rawReportText').textContent;
      navigator.clipboard.writeText(text);
      alert('Report copied to clipboard.');
    }

    // -------------------------------------------------------------
    // VOICE RECOGNITION LOGIC
    // -------------------------------------------------------------
    async function loadSentences() {
      try {
        const res = await fetch('/api/recognition/sentences');
        sentencesList = await res.json();
        if (sentencesList.length > 0) {
          currentSentenceIndex = 0;
          displaySentence();
        }
      } catch (e) {
        console.error(e);
      }
    }

    function displaySentence() {
      if (!sentencesList.length) return;
      const s = sentencesList[currentSentenceIndex];
      document.getElementById('promptCategory').textContent = s.category;
      document.getElementById('promptText').textContent = `"${s.text}"`;
      document.getElementById('promptFocus').textContent = `Phonetic Focus: ${s.phonetic_focus}`;
    }

    function cycleSentence() {
      if (!sentencesList.length) return;
      currentSentenceIndex = (currentSentenceIndex + 1) % sentencesList.length;
      displaySentence();
    }

    async function loadEnrolledSpeakers() {
      try {
        const res = await fetch('/api/recognition/speakers');
        const speakers = await res.json();
        document.getElementById('enrolledCount').textContent = speakers.length;
        const tbody = document.getElementById('enrolledSpeakersBody');
        tbody.innerHTML = '';
        if (speakers.length === 0) {
          tbody.innerHTML = '<tr><td colspan="5" class="px-3 py-3 text-center text-zinc-500 italic">No speakers enrolled yet. Use Mode 2 to register a voice profile.</td></tr>';
          return;
        }
        speakers.forEach(s => {
          const tr = document.createElement('tr');
          tr.className = 'hover:bg-zinc-900/60 transition';
          tr.innerHTML = `
            <td class="px-3 py-2 text-zinc-100 font-semibold">${s.name}</td>
            <td class="px-3 py-2 text-zinc-400">${s.enrolled_at || ''}</td>
            <td class="px-3 py-2 text-zinc-300 font-mono">${s.median_f0 ? s.median_f0.toFixed(1) + ' Hz' : 'N/A'}</td>
            <td class="px-3 py-2 text-zinc-400 font-mono">${s.sample_file || ''}</td>
            <td class="px-3 py-2 text-right">
              <button onclick="deleteSpeaker('${encodeURIComponent(s.name)}')" class="text-zinc-500 hover:text-red-400 text-xs underline font-mono">Delete</button>
            </td>
          `;
          tbody.appendChild(tr);
        });
      } catch (e) {
        console.error(e);
      }
    }

    async function deleteSpeaker(name) {
      if (!confirm(`Delete voice profile for '${decodeURIComponent(name)}'?`)) return;
      await fetch(`/api/recognition/speaker?name=${name}`, { method: 'DELETE' });
      await loadEnrolledSpeakers();
    }

    // -------------------------------------------------------------
    // MICROPHONE CAPTURE & WAV CONVERTER
    // -------------------------------------------------------------
    function toggleRecordDrawer() {
      const drawer = document.getElementById('recordingDrawer');
      drawer.classList.toggle('hidden');
      if (drawer.classList.contains('hidden') && mediaRecorder && mediaRecorder.state === 'recording') {
        stopMicRecording();
      }
    }

    async function startMicRecording() {
      await startGenericAudioStream();
      document.getElementById('btnStartRec').disabled = true;
      document.getElementById('btnStartRec').classList.add('opacity-50');
      document.getElementById('btnStopRec').disabled = false;
      document.getElementById('btnStopRec').classList.remove('opacity-50', 'cursor-not-allowed');
      document.getElementById('recStatusLabel').textContent = 'Recording in progress...';
    }

    async function stopMicRecording() {
      const wavBlob = await stopGenericAudioStream();
      document.getElementById('btnStartRec').disabled = false;
      document.getElementById('btnStartRec').classList.remove('opacity-50');
      document.getElementById('btnStopRec').textContent = 'Stop & Analyze';
      document.getElementById('recStatusLabel').textContent = 'Analyzed & Saved';

      const filename = `recording_${new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)}.wav`;
      await uploadAndAnalyzeAudioBlob(wavBlob, filename);
    }

    async function startRecognitionRecord() {
      const btn = document.getElementById('btnRecogStart');
      const timer = document.getElementById('recogTimer');
      if (mediaRecorder && mediaRecorder.state === 'recording') {
        btn.textContent = 'Processing & Identifying...';
        btn.disabled = true;
        const wavBlob = await stopGenericAudioStream();
        timer.classList.add('hidden');
        btn.textContent = 'Record & Identify Voice';
        btn.disabled = false;
        await identifyVoiceBlob(wavBlob);
      } else {
        await startGenericAudioStream(timer);
        btn.textContent = 'Stop & Identify Speaker';
        btn.className = 'w-full bg-red-600 hover:bg-red-700 text-white font-semibold text-xs py-2 rounded transition flex items-center justify-center gap-2';
      }
    }

    async function startEnrollRecord() {
      const name = document.getElementById('speakerNameInput').value.trim();
      if (!name) {
        alert('Please enter a speaker name before enrolling.');
        return;
      }

      const btn = document.getElementById('btnEnrollStart');
      const timer = document.getElementById('enrollTimer');

      if (mediaRecorder && mediaRecorder.state === 'recording') {
        btn.textContent = 'Enrolling Profile...';
        btn.disabled = true;
        const wavBlob = await stopGenericAudioStream();
        timer.classList.add('hidden');
        btn.textContent = 'Record & Enroll Voice';
        btn.disabled = false;
        await enrollVoiceBlob(wavBlob, name);
      } else {
        await startGenericAudioStream(timer);
        btn.textContent = 'Stop & Save Voice Profile';
        btn.className = 'w-full bg-red-600 hover:bg-red-700 text-white font-semibold text-xs py-2 rounded transition flex items-center justify-center gap-2';
      }
    }

    async function startGenericAudioStream(timerElem = null) {
      mediaStream = await navigator.mediaDevices.getUserMedia({ audio: { sampleRate: 16000, channelCount: 1 } });
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const source = audioContext.createMediaStreamSource(mediaStream);
      micAnalyser = audioContext.createAnalyser();
      micAnalyser.fftSize = 256;
      source.connect(micAnalyser);

      const canvas = document.getElementById('micCanvas');
      if (canvas) drawMicVisualizer();

      recordedChunks = [];
      mediaRecorder = new MediaRecorder(mediaStream);
      mediaRecorder.ondataavailable = e => { if (e.data.size > 0) recordedChunks.push(e.data); };
      mediaRecorder.start(100);

      recordSeconds = 0;
      if (timerElem) {
        timerElem.classList.remove('hidden');
        timerElem.textContent = 'REC 00:00';
      }
      clearInterval(recordInterval);
      recordInterval = setInterval(() => {
        recordSeconds++;
        const m = String(Math.floor(recordSeconds / 60)).padStart(2, '0');
        const s = String(recordSeconds % 60).padStart(2, '0');
        const tStr = `${m}:${s}`;
        document.getElementById('recTimer').textContent = tStr;
        if (timerElem) timerElem.textContent = `REC ${tStr}`;
      }, 1000);
    }

    async function stopGenericAudioStream() {
      clearInterval(recordInterval);
      if (animFrame) cancelAnimationFrame(animFrame);

      return new Promise(resolve => {
        mediaRecorder.onstop = async () => {
          const rawBlob = new Blob(recordedChunks, { type: mediaRecorder.mimeType });
          mediaStream.getTracks().forEach(t => t.stop());

          const arrayBuf = await rawBlob.arrayBuffer();
          const audioBuffer = await audioContext.decodeAudioData(arrayBuf);
          const wavBlob = audioBufferToWav(audioBuffer);
          resolve(wavBlob);
        };
        mediaRecorder.stop();
      });
    }

    function drawMicVisualizer() {
      const canvas = document.getElementById('micCanvas');
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      const bufferLength = micAnalyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);

      function draw() {
        animFrame = requestAnimationFrame(draw);
        micAnalyser.getByteTimeDomainData(dataArray);

        ctx.fillStyle = '#09090b';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        ctx.lineWidth = 1.5;
        ctx.strokeStyle = '#f4f4f5';
        ctx.beginPath();

        const sliceWidth = canvas.width / bufferLength;
        let x = 0;
        for (let i = 0; i < bufferLength; i++) {
          const v = dataArray[i] / 128.0;
          const y = (v * canvas.height) / 2;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
          x += sliceWidth;
        }
        ctx.lineTo(canvas.width, canvas.height / 2);
        ctx.stroke();
      }
      draw();
    }

    function audioBufferToWav(buffer) {
      const numChannels = 1;
      const sampleRate = buffer.sampleRate;
      const format = 1;
      const bitDepth = 16;
      const channelData = buffer.getChannelData(0);
      const dataLength = channelData.length * (bitDepth / 8);
      const bufferLength = 44 + dataLength;

      const arrayBuffer = new ArrayBuffer(bufferLength);
      const view = new DataView(arrayBuffer);

      function writeStr(offset, str) {
        for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
      }

      writeStr(0, 'RIFF');
      view.setUint32(4, 36 + dataLength, true);
      writeStr(8, 'WAVE');
      writeStr(12, 'fmt ');
      view.setUint32(16, 16, true);
      view.setUint16(20, format, true);
      view.setUint16(22, numChannels, true);
      view.setUint32(24, sampleRate, true);
      view.setUint32(28, sampleRate * numChannels * (bitDepth / 8), true);
      view.setUint16(32, numChannels * (bitDepth / 8), true);
      view.setUint16(34, bitDepth, true);
      writeStr(36, 'data');
      view.setUint32(40, dataLength, true);

      let offset = 44;
      for (let i = 0; i < channelData.length; i++) {
        let sample = Math.max(-1, Math.min(1, channelData[i]));
        sample = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
        view.setInt16(offset, sample, true);
        offset += 2;
      }
      return new Blob([view], { type: 'audio/wav' });
    }

    async function handleFileUpload(event) {
      const file = event.target.files[0];
      if (!file) return;
      await uploadAndAnalyzeAudioBlob(file, file.name);
    }

    async function uploadAndAnalyzeAudioBlob(blob, filename) {
      try {
        const res = await fetch('/api/upload', {
          method: 'POST',
          headers: { 'X-Filename': filename },
          body: blob
        });
        const result = await res.json();
        await refreshFileList();
        document.getElementById('fileSelect').value = result.filename;
        updateActiveAudio();
        currentProfile = result.profile;
        populateDashboard(result.profile);
        loadDatasetTable();
      } catch (err) {
        alert('Upload/Analysis error: ' + err);
      }
    }

    async function identifyVoiceBlob(blob) {
      const card = document.getElementById('recogResultCard');
      card.classList.remove('hidden');

      try {
        const res = await fetch('/api/recognition/identify', {
          method: 'POST',
          body: blob
        });
        const result = await res.json();

        document.getElementById('recogSpeakerName').textContent = result.speaker;
        document.getElementById('recogScore').textContent = `${result.confidence_percent}%`;
        document.getElementById('recogMessage').textContent = result.message;

        const badge = document.getElementById('recogMatchBadge');
        if (result.matched) {
          badge.textContent = 'AUTHENTICATED';
          badge.className = 'text-xs font-mono px-2 py-0.5 rounded border border-emerald-500/40 text-emerald-400 font-bold';
        } else {
          badge.textContent = 'UNVERIFIED';
          badge.className = 'text-xs font-mono px-2 py-0.5 rounded border border-amber-500/40 text-amber-400 font-bold';
        }

        const tbody = document.getElementById('recogCandidatesBody');
        tbody.innerHTML = '';
        (result.candidates || []).forEach(c => {
          const tr = document.createElement('tr');
          tr.className = 'hover:bg-zinc-900/60 transition';
          tr.innerHTML = `
            <td class="px-3 py-2 text-zinc-100 font-semibold">${c.name}</td>
            <td class="px-3 py-2">${c.similarity}</td>
            <td class="px-3 py-2 font-bold">${c.confidence_percent}%</td>
            <td class="px-3 py-2 text-zinc-400">${c.meta && c.meta.median_f0 ? c.meta.median_f0.toFixed(1) + ' Hz' : 'N/A'}</td>
            <td class="px-3 py-2">${c.similarity >= (result.threshold || 0.75) ? '<span class="text-emerald-400 font-semibold">Match</span>' : '<span class="text-zinc-500">Below threshold</span>'}</td>
          `;
          tbody.appendChild(tr);
        });

      } catch (err) {
        alert('Recognition error: ' + err);
      }
    }

    async function enrollVoiceBlob(blob, name) {
      const sentenceText = sentencesList[currentSentenceIndex] ? sentencesList[currentSentenceIndex].text : '';
      try {
        const res = await fetch(`/api/recognition/enroll?name=${encodeURIComponent(name)}`, {
          method: 'POST',
          headers: { 'X-Sentence': encodeURIComponent(sentenceText) },
          body: blob
        });
        const result = await res.json();
        alert(`Speaker '${name}' successfully enrolled into biometric database!`);
        document.getElementById('speakerNameInput').value = '';
        await loadEnrolledSpeakers();
        await refreshFileList();
      } catch (err) {
        alert('Enrollment error: ' + err);
      }
    }

    window.addEventListener('DOMContentLoaded', initApp);
  </script>
</body>
</html>
"""


class DashboardHTTPRequestHandler(BaseHTTPRequestHandler):

    def send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        params = urllib.parse.parse_qs(parsed.query)

        # 1. Main Dashboard HTML
        if path == "/" or path == "/index.html":
            body = HTML_CONTENT.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        # 2. List available audio files
        if path == "/api/files":
            files = get_available_audio_files()
            self.send_json(files)
            return

        # 3. Stream audio file
        if path == "/api/audio":
            filename = params.get("file", ["voice.wav"])[0]
            file_path = os.path.join(WORKSPACE_DIR, filename)
            if not os.path.isfile(file_path):
                self.send_error(404, "Audio file not found")
                return
            with open(file_path, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self.wfile.write(data)
            return

        # 4. Run Analysis
        if path == "/api/analyze":
            filename = params.get("file", ["voice.wav"])[0]
            file_path = os.path.join(WORKSPACE_DIR, filename)
            if not os.path.isfile(file_path):
                self.send_error(404, "File not found")
                return
            profile, flat = analyze_audio_signal(file_path)
            report_text = generate_voice_diagnostic_report(profile)
            profile["report_text"] = report_text

            csv_path = os.path.join(WORKSPACE_DIR, "voice_dataset.csv")
            save_rows_to_csv([flat], csv_path)

            self.send_json(profile)
            return

        # 5. Dedicated Individual Analysis Plots API
        # Path format: /api/plot/<plot_type>?file=voice.wav
        if path.startswith("/api/plot/"):
            plot_type = path.split("/api/plot/")[1]
            filename = params.get("file", ["voice.wav"])[0]
            download = params.get("download", ["0"])[0] == "1"
            file_path = os.path.join(WORKSPACE_DIR, filename)
            if not os.path.isfile(file_path):
                self.send_error(404, "Audio file not found")
                return

            base_name = os.path.splitext(filename)[0]
            output_img = os.path.join(WORKSPACE_DIR, f"temp_{plot_type}_{base_name}.png")

            try:
                if plot_type == "dft":
                    render_dft_plot(file_path, save_path=output_img)
                elif plot_type == "spectrogram":
                    render_spectrogram_plot(file_path, save_path=output_img)
                elif plot_type == "pitch":
                    render_pitch_plot(file_path, save_path=output_img)
                elif plot_type == "lpc":
                    render_lpc_plot(file_path, save_path=output_img)
                elif plot_type == "mfcc":
                    render_mfcc_plot(file_path, save_path=output_img)
                elif plot_type == "vowel_space":
                    render_vowel_space_plot(file_path, save_path=output_img)
                elif plot_type == "glottal":
                    render_glottal_plot(file_path, save_path=output_img)
                elif plot_type == "composite":
                    render_voice_dashboard(file_path, save_path=output_img, show=False)
                else:
                    self.send_error(400, "Unknown plot type")
                    return

                with open(output_img, "rb") as f:
                    data = f.read()

                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(data)))
                if download:
                    self.send_header("Content-Disposition", f'attachment; filename="{plot_type}_{base_name}.png"')
                else:
                    self.send_header("Content-Disposition", "inline")
                self.end_headers()
                self.wfile.write(data)
                return
            except Exception as e:
                self.send_error(500, f"Error rendering plot: {e}")
                return

        # 6. Raw DFT Spectrum Data Points Export (CSV)
        if path == "/api/data/dft_csv":
            filename = params.get("file", ["voice.wav"])[0]
            file_path = os.path.join(WORKSPACE_DIR, filename)
            if not os.path.isfile(file_path):
                self.send_error(404, "Audio file not found")
                return

            base_name = os.path.splitext(filename)[0]
            temp_csv = os.path.join(WORKSPACE_DIR, f"temp_dft_{base_name}.csv")
            temp_png = os.path.join(WORKSPACE_DIR, f"temp_dft_{base_name}.png")

            render_dft_plot(file_path, save_path=temp_png, csv_save_path=temp_csv)

            with open(temp_csv, "rb") as f:
                data = f.read()

            self.send_response(200)
            self.send_header("Content-Type", "text/csv")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Content-Disposition", f'attachment; filename="dft_spectrum_{base_name}.csv"')
            self.end_headers()
            self.wfile.write(data)

            try:
                os.remove(temp_csv)
                os.remove(temp_png)
            except Exception:
                pass
            return

        # 7. Complete ZIP Bundle Export
        if path == "/api/export/zip":
            filename = params.get("file", ["voice.wav"])[0]
            file_path = os.path.join(WORKSPACE_DIR, filename)
            if not os.path.isfile(file_path):
                self.send_error(404, "Audio file not found")
                return

            base_name = os.path.splitext(filename)[0]
            temp_zip = os.path.join(WORKSPACE_DIR, f"temp_bundle_{base_name}.zip")

            export_all_analyses_bundle(file_path, temp_zip)

            with open(temp_zip, "rb") as f:
                data = f.read()

            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Content-Disposition", f'attachment; filename="analyses_bundle_{base_name}.zip"')
            self.end_headers()
            self.wfile.write(data)

            try:
                os.remove(temp_zip)
            except Exception:
                pass
            return

        # 8. Serve Image by File Name
        if path == "/api/image":
            filename = params.get("file", ["voice_dashboard.png"])[0]
            file_path = os.path.join(WORKSPACE_DIR, filename)
            if not os.path.isfile(file_path):
                file_path = os.path.join(WORKSPACE_DIR, "assets", filename)
            if not os.path.isfile(file_path):
                self.send_error(404, "Image not found")
                return
            with open(file_path, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        # 9. Dataset Rows
        if path == "/api/dataset":
            records = load_dataset_records()
            self.send_json(records)
            return

        # 10. Raw CSV Download
        if path == "/voice_dataset.csv":
            csv_path = os.path.join(WORKSPACE_DIR, "voice_dataset.csv")
            if not os.path.isfile(csv_path):
                self.send_error(404, "Dataset not found")
                return
            with open(csv_path, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/csv")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        # 11. Voice Recognition: Sentences & Speakers
        if path == "/api/recognition/sentences":
            sentences = get_all_sentences()
            self.send_json(sentences)
            return

        if path == "/api/recognition/speakers":
            speakers = default_db.list_speakers()
            self.send_json(speakers)
            return

        self.send_error(404, "Not found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        params = urllib.parse.parse_qs(parsed.query)

        # 1. Upload for General DSP Analysis
        if path == "/api/upload":
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length == 0:
                self.send_error(400, "Empty payload")
                return

            body = self.rfile.read(content_length)
            raw_filename = self.headers.get("X-Filename", "")
            if raw_filename:
                filename = os.path.basename(urllib.parse.unquote(raw_filename))
            else:
                filename = f"recording_{int(time.time())}.wav"

            if not filename.lower().endswith(".wav"):
                filename += ".wav"

            file_path = os.path.join(WORKSPACE_DIR, filename)
            with open(file_path, "wb") as f:
                f.write(body)

            try:
                profile, flat = analyze_audio_signal(file_path)
                report_text = generate_voice_diagnostic_report(profile)
                profile["report_text"] = report_text

                csv_path = os.path.join(WORKSPACE_DIR, "voice_dataset.csv")
                save_rows_to_csv([flat], csv_path)

                save_img = os.path.join(WORKSPACE_DIR, "voice_dashboard.png")
                try:
                    render_voice_dashboard(file_path, save_img, show=False)
                except Exception:
                    pass

                self.send_json({
                    "success": True,
                    "filename": filename,
                    "profile": profile
                })
            except Exception as e:
                self.send_json({"success": False, "error": str(e)}, status=500)
            return

        # 2. Voice Recognition: Enroll Speaker
        if path == "/api/recognition/enroll":
            name = params.get("name", [""])[0]
            if not name:
                self.send_error(400, "Missing speaker name")
                return

            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            sentence_text = urllib.parse.unquote(self.headers.get("X-Sentence", ""))
            safe_name = "".join(c for c in name if c.isalnum() or c in (" ", "_", "-")).strip()
            filename = f"speaker_{safe_name.replace(' ', '_')}_{int(time.time())}.wav"
            file_path = os.path.join(WORKSPACE_DIR, filename)

            with open(file_path, "wb") as f:
                f.write(body)

            try:
                enrolled = default_db.enroll_speaker(name, file_path, sentence_text)
                self.send_json({"success": True, "speaker": enrolled})
            except Exception as e:
                self.send_json({"success": False, "error": str(e)}, status=500)
            return

        # 3. Voice Recognition: Identify Speaker
        if path == "/api/recognition/identify":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            temp_file = os.path.join(WORKSPACE_DIR, f"temp_recog_{int(time.time())}.wav")
            with open(temp_file, "wb") as f:
                f.write(body)

            try:
                result = default_db.identify_speaker(temp_file)
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                self.send_json(result)
            except Exception as e:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                self.send_json({"matched": False, "error": str(e)}, status=500)
            return

        self.send_error(404, "Endpoint not found")

    def do_DELETE(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        params = urllib.parse.parse_qs(parsed.query)

        if path == "/api/recognition/speaker":
            name = params.get("name", [""])[0]
            if not name:
                self.send_error(400, "Missing speaker name")
                return
            deleted = default_db.delete_speaker(urllib.parse.unquote(name))
            self.send_json({"success": deleted})
            return

        self.send_error(404, "Endpoint not found")

    def log_message(self, format, *args):
        return


def start_server(port=PORT, open_browser=True):
    server = HTTPServer(("127.0.0.1", port), DashboardHTTPRequestHandler)
    url = f"http://localhost:{port}"
    print("=" * 65)
    print(f"  vox-dsp Centralized Dashboard & Speaker Recognition Studio")
    print(f"  URL: {url}")
    print("  Press Ctrl+C to stop the dashboard server.")
    print("=" * 65)

    if open_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down dashboard server...")
        server.server_close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="vox-dsp Centralized Dashboard & Recognition Server")
    parser.add_argument("--port", type=int, default=PORT, help="Port to listen on (default: 8050)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    args = parser.parse_args()

    start_server(port=args.port, open_browser=not args.no_browser)
