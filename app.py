"""
vox-dsp Minimalist Centralized Access Dashboard & Live Audio Recording Server.
Zero-dependency HTTP server using Python's standard library.
Features:
- Minimalist Swiss/monochrome design with high data density
- In-browser live microphone audio recording & real-time analysis
- File upload for external .wav recordings
- Interactive audio playback, 6-panel visual dashboard, and dataset explorer
"""

import os
import sys
import json
import glob
import time
import urllib.parse
import mimetypes
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
import soundfile as sf
import numpy as np

from analyze_voice import analyze_audio_signal, save_rows_to_csv, load_audio
from audio_dsp.diagnostic_engine import generate_voice_diagnostic_report
from dashboard import render_voice_dashboard

PORT = 8050
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_available_audio_files():
    wavs = glob.glob(os.path.join(WORKSPACE_DIR, "*.wav"))
    # Return sorted by modification time (most recent first)
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
  <title>vox-dsp | Voice Acoustic Biometrics</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {
      darkMode: 'class',
      theme: {
        extend: {
          fontFamily: {
            sans: ['-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'Helvetica Neue', 'Arial', 'sans-serif'],
            mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', 'monospace']
          }
        }
      }
    }
  </script>
  <style>
    /* Minimalist styling */
    body { background-color: #09090b; color: #f4f4f5; }
    .card-surface { background-color: #121215; border: 1px solid #27272a; }
    .tab-active { border-bottom: 2px solid #f4f4f5; color: #f4f4f5; font-weight: 500; }
    .status-dot { width: 6px; height: 6px; border-radius: 50%; display: inline-block; }
    .pulse-rec { animation: pulse 1.2s infinite ease-in-out; }
    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.3; transform: scale(1.2); }
    }
  </style>
</head>
<body class="min-h-screen flex flex-col font-sans antialiased text-sm text-zinc-300">

  <!-- Minimal Header -->
  <header class="border-b border-zinc-800/80 bg-zinc-950/80 backdrop-blur sticky top-0 z-50">
    <div class="max-w-6xl mx-auto px-4 h-14 flex items-center justify-between">
      <div class="flex items-center space-x-3">
        <span class="font-mono text-xs tracking-widest uppercase text-zinc-400 font-semibold">vox-dsp</span>
        <span class="text-zinc-600">/</span>
        <span class="text-xs text-zinc-400">Acoustic Biometrics</span>
      </div>

      <!-- Action Controls -->
      <div class="flex items-center space-x-2">
        <select id="fileSelect" class="bg-zinc-900 border border-zinc-800 text-xs rounded px-2.5 py-1 text-zinc-300 focus:outline-none focus:border-zinc-600 font-mono">
          <option value="voice.wav">voice.wav</option>
        </select>

        <button onclick="toggleRecordDrawer()" id="btnRecordToggle" class="border border-zinc-800 hover:border-zinc-600 bg-zinc-900 text-xs px-3 py-1 rounded text-zinc-200 transition flex items-center gap-1.5 font-medium">
          <span class="w-2 h-2 rounded-full bg-red-500"></span>
          <span>Record Voice</span>
        </button>

        <label class="border border-zinc-800 hover:border-zinc-600 bg-zinc-900 text-xs px-3 py-1 rounded text-zinc-200 transition cursor-pointer font-medium">
          <span>Upload WAV</span>
          <input type="file" id="fileUploadInput" accept=".wav" class="hidden" onchange="handleFileUpload(event)">
        </label>

        <button id="btnAnalyze" onclick="triggerAnalyze()" class="bg-zinc-100 hover:bg-white text-zinc-950 font-semibold text-xs px-3.5 py-1 rounded transition shadow-sm">
          Run Analysis
        </button>
      </div>
    </div>
  </header>

  <!-- Live Recording Drawer (Hidden by default) -->
  <div id="recordingDrawer" class="hidden border-b border-zinc-800 bg-zinc-900/90 py-4 px-4 transition">
    <div class="max-w-6xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
      <div class="flex items-center space-x-3">
        <span id="recDot" class="w-2.5 h-2.5 rounded-full bg-red-500 pulse-rec"></span>
        <span id="recStatusLabel" class="text-xs uppercase tracking-wider font-mono text-red-400 font-semibold">Microphone Ready</span>
        <span id="recTimer" class="font-mono text-sm font-bold text-zinc-100 bg-zinc-950 px-2 py-0.5 rounded border border-zinc-800">00:00</span>
      </div>

      <!-- Minimal Live Waveform Canvas -->
      <canvas id="micCanvas" width="280" height="32" class="rounded bg-zinc-950 border border-zinc-800/80"></canvas>

      <div class="flex items-center space-x-2">
        <button id="btnStartRec" onclick="startMicRecording()" class="bg-red-500 hover:bg-red-600 text-white text-xs px-3 py-1 rounded font-medium transition">
          Start Recording
        </button>
        <button id="btnStopRec" onclick="stopMicRecording()" disabled class="bg-zinc-800 hover:bg-zinc-700 text-zinc-300 text-xs px-3 py-1 rounded font-medium transition opacity-50 cursor-not-allowed">
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
        <span class="text-zinc-500">ACTIVE:</span>
        <span id="activeFileLabel" class="text-zinc-200">voice.wav</span>
        <span id="activeDurationLabel" class="text-zinc-500">6.59s @ 16kHz</span>
      </div>
      <audio id="audioPlayer" controls class="h-7 w-72 rounded opacity-90"></audio>
      <div class="flex items-center space-x-2">
        <button onclick="renderDashboardImg()" class="border border-zinc-800 hover:border-zinc-700 text-zinc-400 hover:text-zinc-200 px-2.5 py-1 rounded text-xs transition">
          View 6-Panel Plot
        </button>
        <button onclick="exportJSON()" class="border border-zinc-800 hover:border-zinc-700 text-zinc-400 hover:text-zinc-200 px-2.5 py-1 rounded text-xs transition">
          Export JSON
        </button>
      </div>
    </div>
  </div>

  <!-- Main Content -->
  <main class="max-w-6xl w-full mx-auto px-4 py-6 flex-1 space-y-6">

    <!-- 4-Card Minimal Grid -->
    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
      
      <!-- Card 1: Morphology -->
      <div class="card-surface rounded-lg p-4">
        <div class="text-[11px] font-mono uppercase tracking-wider text-zinc-500">01 / Speaker Morphology</div>
        <div class="mt-2 text-base font-semibold text-zinc-100" id="cardSpeaker">Adult Male</div>
        <div class="mt-2 text-xs text-zinc-400 space-y-0.5">
          <div class="flex justify-between"><span>Vocal Tract Length:</span> <strong id="cardVTL" class="text-zinc-200 font-mono">17.2 cm</strong></div>
          <div class="flex justify-between"><span>LPC Model Order:</span> <strong id="cardLPCOrder" class="text-zinc-200 font-mono">18</strong></div>
        </div>
      </div>

      <!-- Card 2: Intonation -->
      <div class="card-surface rounded-lg p-4">
        <div class="text-[11px] font-mono uppercase tracking-wider text-zinc-500">02 / Glottal Pitch (f0)</div>
        <div class="mt-2 text-base font-semibold text-zinc-100 flex items-baseline gap-1.5">
          <span id="cardF0" class="text-lg font-mono">117.1</span>
          <span class="text-xs text-zinc-500 font-normal">Hz median</span>
        </div>
        <div class="mt-2 text-xs text-zinc-400 space-y-0.5">
          <div class="flex justify-between"><span>Prosodic Range:</span> <strong id="cardSpread" class="text-zinc-200 font-mono">11.45 st</strong></div>
          <div class="flex justify-between"><span>Voiced Activity:</span> <strong id="cardDuty" class="text-zinc-200 font-mono">23.0%</strong></div>
        </div>
      </div>

      <!-- Card 3: Formants -->
      <div class="card-surface rounded-lg p-4">
        <div class="text-[11px] font-mono uppercase tracking-wider text-zinc-500">03 / Formant Resonances</div>
        <div class="mt-2 text-xs font-mono text-zinc-200 font-semibold" id="cardFormants">F1: 435 | F2: 1595 Hz</div>
        <div class="mt-2 text-xs text-zinc-400 space-y-0.5">
          <div class="flex justify-between"><span>Dominant Vowel:</span> <strong id="cardVowel" class="text-zinc-200 truncate max-w-[120px] text-right">Mid-Central</strong></div>
          <div class="flex justify-between"><span>Vowel Space Area:</span> <strong id="cardVSA" class="text-zinc-200 font-mono">111,332 Hz²</strong></div>
        </div>
      </div>

      <!-- Card 4: Biometrics -->
      <div class="card-surface rounded-lg p-4">
        <div class="flex items-center justify-between text-[11px] font-mono uppercase tracking-wider text-zinc-500">
          <span>04 / Voice Stability</span>
          <span id="cardStatusBadge" class="text-[10px] font-mono px-1.5 py-0.5 rounded border border-zinc-700 text-zinc-300">BORDERLINE</span>
        </div>
        <div class="mt-2 text-base font-semibold text-zinc-100 flex items-baseline gap-1.5">
          <span id="cardJitter" class="text-lg font-mono">1.29%</span>
          <span class="text-xs text-zinc-500 font-normal">jitter (local)</span>
        </div>
        <div class="mt-2 text-xs text-zinc-400 space-y-0.5">
          <div class="flex justify-between"><span>Harmonicity (HNR):</span> <strong id="cardHNR" class="text-zinc-200 font-mono">10.99 dB</strong></div>
          <div class="flex justify-between"><span>Continuous CPPS:</span> <strong id="cardCPPS" class="text-zinc-200 font-mono">34.93 dB</strong></div>
        </div>
      </div>

    </div>

    <!-- Minimal Navigation Tabs -->
    <div class="border-b border-zinc-800">
      <nav class="flex space-x-6 text-xs font-mono uppercase tracking-wider">
        <button onclick="switchTab('report')" id="tabBtn-report" class="py-2.5 tab-active">
          Clinical Evaluation
        </button>
        <button onclick="switchTab('visual')" id="tabBtn-visual" class="py-2.5 text-zinc-500 hover:text-zinc-300">
          6-Panel Diagnostic Visualizer
        </button>
        <button onclick="switchTab('dataset')" id="tabBtn-dataset" class="py-2.5 text-zinc-500 hover:text-zinc-300">
          Dataset Records (<span id="datasetCount">1</span>)
        </button>
      </nav>
    </div>

    <!-- Tab 1: Clinical Evaluation -->
    <div id="tabContent-report" class="space-y-4">
      <div class="card-surface rounded-lg p-5 space-y-4">
        <div class="flex items-center justify-between border-b border-zinc-800/80 pb-3">
          <div class="text-xs font-mono uppercase tracking-wider text-zinc-400 font-semibold">ASHA / Praat Normative Comparison</div>
          <span id="evalOverallHealth" class="text-xs text-zinc-400 font-mono">Evaluated vs standard acoustic criteria</span>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
          <div class="space-y-2 border border-zinc-800/80 rounded p-3 bg-zinc-950/40">
            <div class="text-[11px] font-mono uppercase text-zinc-500 font-semibold">Vocal Stability Thresholds</div>
            <div class="flex justify-between py-1 border-b border-zinc-900"><span>Jitter (local, %):</span><span id="repJitter" class="font-mono text-zinc-200">1.29%</span></div>
            <div class="flex justify-between py-1 border-b border-zinc-900"><span>Shimmer (local, %):</span><span id="repShimmer" class="font-mono text-zinc-200">14.75%</span></div>
            <div class="flex justify-between py-1 border-b border-zinc-900"><span>Harmonics-to-Noise Ratio:</span><span id="repHNR" class="font-mono text-zinc-200">10.99 dB</span></div>
            <div class="flex justify-between py-1"><span>Continuous CPPS:</span><span id="repCPPS" class="font-mono text-zinc-200">34.93 dB [Pass]</span></div>
          </div>

          <div class="space-y-2 border border-zinc-800/80 rounded p-3 bg-zinc-950/40">
            <div class="text-[11px] font-mono uppercase text-zinc-500 font-semibold">Resonance & Glottal Source</div>
            <div class="flex justify-between py-1 border-b border-zinc-900"><span>Glottal Flow NAQ:</span><span id="repNAQ" class="font-mono text-zinc-200">0.092</span></div>
            <div class="flex justify-between py-1 border-b border-zinc-900"><span>Vocal Tract Length:</span><span id="repVTL" class="font-mono text-zinc-200">17.2 cm</span></div>
            <div class="flex justify-between py-1 border-b border-zinc-900"><span>Prosodic Style:</span><span id="repProsody" class="font-mono text-zinc-200">Dynamic</span></div>
            <div class="flex justify-between py-1"><span>Phonetic VSA:</span><span id="repVSA" class="font-mono text-zinc-200">111,332 Hz²</span></div>
          </div>
        </div>

        <div>
          <div class="flex items-center justify-between mb-2">
            <span class="text-[11px] font-mono uppercase tracking-wider text-zinc-500">Narrative Synthesis</span>
            <button onclick="copyReportText()" class="text-xs text-zinc-400 hover:text-zinc-200 underline font-mono">Copy Report</button>
          </div>
          <pre id="rawReportText" class="p-3.5 rounded bg-zinc-950 border border-zinc-800 font-mono text-xs text-zinc-300 leading-relaxed overflow-x-auto"></pre>
        </div>
      </div>
    </div>

    <!-- Tab 2: 6-Panel Diagnostic Visualizer -->
    <div id="tabContent-visual" class="hidden space-y-3">
      <div class="card-surface rounded-lg p-4 flex flex-col items-center">
        <div class="w-full flex justify-between items-center mb-3">
          <span class="text-xs font-mono uppercase tracking-wider text-zinc-400 font-semibold">6-Panel High-Resolution Acoustic Dashboard</span>
          <a id="btnDownloadImg" href="/api/image?file=voice_dashboard.png" download="voice_dashboard.png" class="border border-zinc-800 hover:border-zinc-700 bg-zinc-900 text-xs px-3 py-1 rounded text-zinc-200 transition font-mono">
            Download PNG
          </a>
        </div>
        <div class="w-full border border-zinc-800 rounded bg-zinc-950 p-1 flex justify-center">
          <img id="dashboardImg" src="/api/image?file=voice_dashboard.png" alt="6-Panel Diagnostic Dashboard" class="max-w-full h-auto rounded">
        </div>
      </div>
    </div>

    <!-- Tab 3: Dataset Archive -->
    <div id="tabContent-dataset" class="hidden space-y-3">
      <div class="card-surface rounded-lg p-4 space-y-3">
        <div class="flex justify-between items-center">
          <span class="text-xs font-mono uppercase tracking-wider text-zinc-400 font-semibold">voice_dataset.csv Archive</span>
          <a href="/voice_dataset.csv" download class="border border-zinc-800 hover:border-zinc-700 text-xs px-2.5 py-1 rounded font-mono text-zinc-300 transition">Download CSV</a>
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

  </main>

  <footer class="border-t border-zinc-800/60 py-3 text-center text-xs font-mono text-zinc-600">
    vox-dsp / minimal acoustic biometrics
  </footer>

  <script>
    let currentProfile = null;
    let mediaStream = null;
    let audioContext = null;
    let mediaRecorder = null;
    let recordedChunks = [];
    let recordInterval = null;
    let recordSeconds = 0;
    let micAnalyser = null;
    let animFrame = null;

    async function loadInitialData() {
      await refreshFileList();
      updateActiveAudio();
      triggerAnalyze();
      loadDatasetTable();
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
    }

    document.getElementById('fileSelect').addEventListener('change', () => {
      updateActiveAudio();
      triggerAnalyze();
    });

    async function triggerAnalyze() {
      const file = document.getElementById('fileSelect').value || 'voice.wav';
      const btn = document.getElementById('btnAnalyze');
      btn.textContent = 'Analyzing...';
      btn.disabled = true;

      try {
        const res = await fetch(`/api/analyze?file=${encodeURIComponent(file)}`);
        const data = await res.json();
        currentProfile = data;
        populateDashboard(data);
      } catch (err) {
        alert('Analysis error: ' + err);
      } finally {
        btn.textContent = 'Run Analysis';
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

      // Cards
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

      const badge = document.getElementById('cardStatusBadge');
      badge.textContent = (diag.jitter_flag || 'NORMAL').toUpperCase();

      // Report Tab
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

    async function renderDashboardImg() {
      const file = document.getElementById('fileSelect').value || 'voice.wav';
      const img = document.getElementById('dashboardImg');
      img.style.opacity = '0.4';
      const res = await fetch(`/api/dashboard?file=${encodeURIComponent(file)}`);
      const data = await res.json();
      img.src = `${data.image_url}&t=${Date.now()}`;
      img.style.opacity = '1.0';
      switchTab('visual');
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

    function switchTab(tabId) {
      ['report', 'visual', 'dataset'].forEach(t => {
        document.getElementById(`tabContent-${t}`).classList.add('hidden');
        document.getElementById(`tabBtn-${t}`).className = 'py-2.5 text-zinc-500 hover:text-zinc-300';
      });
      document.getElementById(`tabContent-${tabId}`).classList.remove('hidden');
      document.getElementById(`tabBtn-${tabId}`).className = 'py-2.5 tab-active';
    }

    function exportJSON() {
      if (!currentProfile) return;
      const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(currentProfile, null, 2));
      const dlAnchor = document.createElement('a');
      dlAnchor.setAttribute("href", dataStr);
      dlAnchor.setAttribute("download", `profile_${currentProfile.file || 'voice'}.json`);
      dlAnchor.click();
    }

    function copyReportText() {
      const text = document.getElementById('rawReportText').textContent;
      navigator.clipboard.writeText(text);
      alert('Report copied to clipboard.');
    }

    // -------------------------------------------------------------
    // LIVE MICROPHONE RECORDING LOGIC
    // -------------------------------------------------------------
    function toggleRecordDrawer() {
      const drawer = document.getElementById('recordingDrawer');
      drawer.classList.toggle('hidden');
      if (drawer.classList.contains('hidden') && mediaRecorder && mediaRecorder.state === 'recording') {
        stopMicRecording();
      }
    }

    async function startMicRecording() {
      try {
        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: { sampleRate: 16000, channelCount: 1 } });
      } catch (err) {
        alert('Microphone access denied or unavailable: ' + err.message);
        return;
      }

      audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const source = audioContext.createMediaStreamSource(mediaStream);
      micAnalyser = audioContext.createAnalyser();
      micAnalyser.fftSize = 256;
      source.connect(micAnalyser);
      drawMicVisualizer();

      recordedChunks = [];
      mediaRecorder = new MediaRecorder(mediaStream);
      mediaRecorder.ondataavailable = e => { if (e.data.size > 0) recordedChunks.push(e.data); };
      mediaRecorder.start(100);

      recordSeconds = 0;
      document.getElementById('recTimer').textContent = '00:00';
      recordInterval = setInterval(() => {
        recordSeconds++;
        const m = String(Math.floor(recordSeconds / 60)).padStart(2, '0');
        const s = String(recordSeconds % 60).padStart(2, '0');
        document.getElementById('recTimer').textContent = `${m}:${s}`;
      }, 1000);

      document.getElementById('btnStartRec').disabled = true;
      document.getElementById('btnStartRec').classList.add('opacity-50');
      document.getElementById('btnStopRec').disabled = false;
      document.getElementById('btnStopRec').classList.remove('opacity-50', 'cursor-not-allowed');
      document.getElementById('recStatusLabel').textContent = 'Recording in progress...';
      document.getElementById('recStatusLabel').className = 'text-xs uppercase tracking-wider font-mono text-red-500 font-semibold';
    }

    async function stopMicRecording() {
      if (!mediaRecorder) return;
      clearInterval(recordInterval);
      if (animFrame) cancelAnimationFrame(animFrame);

      document.getElementById('btnStopRec').textContent = 'Processing...';
      document.getElementById('btnStopRec').disabled = true;

      mediaRecorder.onstop = async () => {
        const rawBlob = new Blob(recordedChunks, { type: mediaRecorder.mimeType });
        mediaStream.getTracks().forEach(t => t.stop());

        // Decode audio to standard 16-bit PCM WAV
        const arrayBuf = await rawBlob.arrayBuffer();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuf);
        const wavBlob = audioBufferToWav(audioBuffer);

        const filename = `recording_${new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)}.wav`;
        await uploadAndAnalyzeAudioBlob(wavBlob, filename);

        // Reset drawer state
        document.getElementById('btnStartRec').disabled = false;
        document.getElementById('btnStartRec').classList.remove('opacity-50');
        document.getElementById('btnStopRec').textContent = 'Stop & Analyze';
        document.getElementById('recStatusLabel').textContent = 'Saved and Analyzed';
        document.getElementById('recStatusLabel').className = 'text-xs uppercase tracking-wider font-mono text-zinc-400 font-semibold';
      };

      mediaRecorder.stop();
    }

    function drawMicVisualizer() {
      const canvas = document.getElementById('micCanvas');
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

    // Helper: Convert AudioBuffer to 16-bit PCM WAV Blob
    function audioBufferToWav(buffer) {
      const numChannels = 1; // mono
      const sampleRate = buffer.sampleRate;
      const format = 1; // PCM
      const bitDepth = 16;
      
      const channelData = buffer.getChannelData(0);
      const dataLength = channelData.length * (bitDepth / 8);
      const bufferLength = 44 + dataLength;
      
      const arrayBuffer = new ArrayBuffer(bufferLength);
      const view = new DataView(arrayBuffer);
      
      // RIFF identifier
      writeString(view, 0, 'RIFF');
      view.setUint32(4, 36 + dataLength, true);
      writeString(view, 8, 'WAVE');
      // fmt sub-chunk
      writeString(view, 12, 'fmt ');
      view.setUint32(16, 16, true);
      view.setUint16(20, format, true);
      view.setUint16(22, numChannels, true);
      view.setUint32(24, sampleRate, true);
      view.setUint32(28, sampleRate * numChannels * (bitDepth / 8), true);
      view.setUint16(32, numChannels * (bitDepth / 8), true);
      view.setUint16(34, bitDepth, true);
      // data sub-chunk
      writeString(view, 36, 'data');
      view.setUint32(40, dataLength, true);
      
      // Write 16-bit samples
      let offset = 44;
      for (let i = 0; i < channelData.length; i++) {
        let sample = Math.max(-1, Math.min(1, channelData[i]));
        sample = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
        view.setInt16(offset, sample, true);
        offset += 2;
      }
      return new Blob([view], { type: 'audio/wav' });
    }

    function writeString(view, offset, string) {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    }

    async function handleFileUpload(event) {
      const file = event.target.files[0];
      if (!file) return;
      await uploadAndAnalyzeAudioBlob(file, file.name);
    }

    async function uploadAndAnalyzeAudioBlob(blob, filename) {
      const formData = new FormData();
      formData.append('audio', blob, filename);

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
        alert('Upload/Analysis failed: ' + err);
      }
    }

    window.addEventListener('DOMContentLoaded', loadInitialData);
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

        # 5. Render 6-Panel Dashboard image
        if path == "/api/dashboard":
            filename = params.get("file", ["voice.wav"])[0]
            file_path = os.path.join(WORKSPACE_DIR, filename)
            save_img = os.path.join(WORKSPACE_DIR, "voice_dashboard.png")
            render_voice_dashboard(file_path, save_img, show=False)
            self.send_json({"image_url": "/api/image?file=voice_dashboard.png"})
            return

        # 6. Serve Image
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

        # 7. Dataset rows
        if path == "/api/dataset":
            records = load_dataset_records()
            self.send_json(records)
            return

        # 8. Raw CSV Download
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

        self.send_error(404, "Not found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        # In-Browser Live Recording / Upload Endpoint
        if path == "/api/upload":
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length == 0:
                self.send_error(400, "Empty payload")
                return

            body = self.rfile.read(content_length)

            # Extract filename from header or generate timestamped name
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

            # Immediate analysis of the recorded audio
            try:
                profile, flat = analyze_audio_signal(file_path)
                report_text = generate_voice_diagnostic_report(profile)
                profile["report_text"] = report_text

                # Update dataset CSV
                csv_path = os.path.join(WORKSPACE_DIR, "voice_dataset.csv")
                save_rows_to_csv([flat], csv_path)

                # Generate updated 6-panel dashboard PNG for the new voice recording
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

        self.send_error(404, "Endpoint not found")

    def log_message(self, format, *args):
        return


def start_server(port=PORT, open_browser=True):
    server = HTTPServer(("127.0.0.1", port), DashboardHTTPRequestHandler)
    url = f"http://localhost:{port}"
    print("=" * 65)
    print(f"  vox-dsp Minimalist Dashboard & Recording Studio running!")
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
    parser = argparse.ArgumentParser(description="vox-dsp Minimalist Dashboard & Recording Server")
    parser.add_argument("--port", type=int, default=PORT, help="Port to listen on (default: 8050)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    args = parser.parse_args()

    start_server(port=args.port, open_browser=not args.no_browser)
