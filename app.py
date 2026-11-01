"""
vox-dsp Centralized Access Dashboard Web Application.
Zero-dependency web server using Python's standard library http.server.
Provides an interactive command center for audio playback, real-time DSP analysis,
diagnostic report rendering, 6-panel visual dashboard generation, and dataset exploration.

Usage:
  python app.py
  python app.py --port 8050 --no-browser
"""

import os
import sys
import json
import glob
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
  <title>vox-dsp | Centralized Voice Biometrics Dashboard</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {
      darkMode: 'class',
      theme: {
        extend: {
          colors: {
            brand: {
              50: '#f0f7ff',
              100: '#e0effe',
              500: '#0284c7',
              600: '#0369a1',
              700: '#075985',
              900: '#0c4a6e',
            }
          }
        }
      }
    }
  </script>
  <style>
    .tab-active { border-bottom: 2px solid #0284c7; color: #0284c7; font-weight: 600; }
    .badge-pass { background-color: #dcfce7; color: #15803d; border: 1px solid #bbf7d0; }
    .badge-borderline { background-color: #fef9c3; color: #a16207; border: 1px solid #fef08a; }
    .badge-attention { background-color: #fee2e2; color: #b91c1c; border: 1px solid #fecaca; }
  </style>
</head>
<body class="bg-slate-900 text-slate-100 min-h-screen flex flex-col font-sans antialiased">

  <!-- Header -->
  <header class="bg-slate-800/80 backdrop-blur border-b border-slate-700/60 sticky top-0 z-50">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
      <div class="flex items-center space-x-3">
        <div class="w-9 h-9 rounded-lg bg-gradient-to-tr from-sky-500 to-indigo-600 flex items-center justify-center font-bold text-white shadow-lg shadow-sky-500/20">
          V
        </div>
        <div>
          <h1 class="text-lg font-bold tracking-tight text-white flex items-center gap-2">
            vox-dsp
            <span class="text-xs font-medium px-2 py-0.5 rounded-full bg-sky-500/10 text-sky-400 border border-sky-500/20">v2.0-Intelligent</span>
          </h1>
          <p class="text-xs text-slate-400">Human Voice Signal Processing & Acoustic Biometrics Command Center</p>
        </div>
      </div>

      <!-- Quick Audio Selector & Run -->
      <div class="flex items-center space-x-3">
        <select id="fileSelect" class="bg-slate-700 border border-slate-600 rounded-lg text-sm px-3 py-1.5 text-slate-200 focus:outline-none focus:ring-2 focus:ring-sky-500">
          <option value="voice.wav">voice.wav</option>
        </select>
        <button id="btnAnalyze" onclick="triggerAnalyze()" class="bg-sky-500 hover:bg-sky-600 text-white font-medium text-sm px-4 py-1.5 rounded-lg shadow-sm transition flex items-center gap-1.5">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
          Run Analysis
        </button>
      </div>
    </div>
  </header>

  <!-- Main Container -->
  <main class="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">

    <!-- Audio Player Bar -->
    <div class="bg-slate-800/60 border border-slate-700/60 rounded-xl p-4 flex flex-wrap items-center justify-between gap-4">
      <div class="flex items-center space-x-3">
        <span class="text-xs uppercase tracking-wider text-slate-400 font-semibold">Active Audio</span>
        <span id="activeFileLabel" class="text-sm font-mono text-sky-400 bg-slate-700/50 px-2 py-1 rounded">voice.wav</span>
      </div>
      <audio id="audioPlayer" controls class="h-9 w-full sm:w-96 rounded"></audio>
      <div class="flex items-center space-x-2">
        <button onclick="renderDashboardImg()" class="bg-slate-700 hover:bg-slate-600 text-xs font-medium px-3 py-2 rounded-lg text-slate-200 transition">
          Render 6-Panel Image
        </button>
        <button onclick="exportJSON()" class="bg-slate-700 hover:bg-slate-600 text-xs font-medium px-3 py-2 rounded-lg text-slate-200 transition">
          Export JSON
        </button>
      </div>
    </div>

    <!-- Metric Cards Grid -->
    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      
      <!-- Card 1: Speaker Morphology -->
      <div class="bg-slate-800/80 border border-slate-700/60 rounded-xl p-4 relative overflow-hidden">
        <div class="flex items-center justify-between text-xs text-slate-400 font-semibold uppercase">
          <span>Speaker Morphology</span>
          <span class="text-sky-400">VTLN</span>
        </div>
        <div class="mt-2 flex items-baseline gap-2">
          <span id="cardSpeaker" class="text-xl font-bold text-white">Adult Male</span>
        </div>
        <div class="mt-2 text-xs text-slate-400 space-y-1">
          <div>Est. Vocal Tract: <span id="cardVTL" class="text-slate-200 font-semibold">17.2 cm</span></div>
          <div>Adapted LPC Order: <span id="cardLPCOrder" class="text-slate-200 font-semibold">18</span></div>
        </div>
      </div>

      <!-- Card 2: Glottal Pitch (f0) -->
      <div class="bg-slate-800/80 border border-slate-700/60 rounded-xl p-4 relative overflow-hidden">
        <div class="flex items-center justify-between text-xs text-slate-400 font-semibold uppercase">
          <span>Glottal Pitch ($f_0$)</span>
          <span class="text-emerald-400">Intonation</span>
        </div>
        <div class="mt-2 flex items-baseline gap-2">
          <span id="cardF0" class="text-2xl font-bold text-white">117.1</span>
          <span class="text-xs text-slate-400">Hz (Median)</span>
        </div>
        <div class="mt-2 text-xs text-slate-400 space-y-1">
          <div>Spread: <span id="cardSpread" class="text-slate-200 font-semibold">11.45 semitones</span></div>
          <div>Duty Cycle: <span id="cardDuty" class="text-slate-200 font-semibold">23.0% voiced</span></div>
        </div>
      </div>

      <!-- Card 3: Formants & Vowel Space -->
      <div class="bg-slate-800/80 border border-slate-700/60 rounded-xl p-4 relative overflow-hidden">
        <div class="flex items-center justify-between text-xs text-slate-400 font-semibold uppercase">
          <span>Formants & Timbre</span>
          <span class="text-amber-400">Resonance</span>
        </div>
        <div class="mt-2 flex items-baseline gap-2">
          <span id="cardFormants" class="text-sm font-mono font-bold text-white">F1: 435 | F2: 1595</span>
        </div>
        <div class="mt-2 text-xs text-slate-400 space-y-1">
          <div>Dominant: <span id="cardVowel" class="text-slate-200 font-semibold truncate block">Mid-Central (/uh/)</span></div>
          <div>VSA Area: <span id="cardVSA" class="text-slate-200 font-semibold">111,332 Hz²</span></div>
        </div>
      </div>

      <!-- Card 4: Clinical Stability (Jitter/HNR) -->
      <div class="bg-slate-800/80 border border-slate-700/60 rounded-xl p-4 relative overflow-hidden">
        <div class="flex items-center justify-between text-xs text-slate-400 font-semibold uppercase">
          <span>Clinical Stability</span>
          <span id="cardOverallBadge" class="text-xs px-2 py-0.5 rounded font-medium badge-borderline">Attention</span>
        </div>
        <div class="mt-2 flex items-baseline gap-2">
          <span id="cardJitter" class="text-xl font-bold text-white">1.29%</span>
          <span class="text-xs text-slate-400">Jitter | Shimmer: <span id="cardShimmer">14.7%</span></span>
        </div>
        <div class="mt-2 text-xs text-slate-400 space-y-1">
          <div>Harmonicity: <span id="cardHNR" class="text-slate-200 font-semibold">10.99 dB</span></div>
          <div>Continuous CPPS: <span id="cardCPPS" class="text-slate-200 font-semibold">34.93 dB</span></div>
        </div>
      </div>

    </div>

    <!-- Navigation Tabs -->
    <div class="border-b border-slate-700/80">
      <nav class="flex space-x-6 text-sm">
        <button onclick="switchTab('report')" id="tabBtn-report" class="py-3 px-1 tab-active flex items-center gap-2">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg>
          Clinical Diagnostic Report
        </button>
        <button onclick="switchTab('visual')" id="tabBtn-visual" class="py-3 px-1 text-slate-400 hover:text-slate-200 flex items-center gap-2">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg>
          6-Panel High-Res Visual Dashboard
        </button>
        <button onclick="switchTab('dataset')" id="tabBtn-dataset" class="py-3 px-1 text-slate-400 hover:text-slate-200 flex items-center gap-2">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 10h18M3 14h18m-9-4v8m-7 0h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg>
          Tabular Dataset Explorer (<span id="datasetCount">1</span>)
        </button>
      </nav>
    </div>

    <!-- Tab 1: Clinical Diagnostic Report -->
    <div id="tabContent-report" class="space-y-4">
      <div class="bg-slate-800/80 border border-slate-700/60 rounded-xl p-5 shadow-sm">
        <h2 class="text-sm font-bold uppercase tracking-wider text-slate-300 mb-3 flex items-center gap-2">
          <span class="w-2.5 h-2.5 rounded-full bg-sky-400 inline-block"></span>
          ASHA & Clinical Normative Evaluation
        </h2>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div class="bg-slate-900/60 p-4 rounded-lg border border-slate-700/40 space-y-2">
            <div class="text-xs font-semibold text-slate-400 uppercase">Acoustic Stability Criteria</div>
            <div class="flex justify-between items-center py-1 border-b border-slate-800">
              <span class="text-slate-300">Jitter (local, %):</span>
              <span id="repJitter" class="font-mono text-slate-100">1.29% [Borderline]</span>
            </div>
            <div class="flex justify-between items-center py-1 border-b border-slate-800">
              <span class="text-slate-300">Shimmer (local, %):</span>
              <span id="repShimmer" class="font-mono text-slate-100">14.75% [Attention]</span>
            </div>
            <div class="flex justify-between items-center py-1 border-b border-slate-800">
              <span class="text-slate-300">HNR (Harmonics-to-Noise):</span>
              <span id="repHNR" class="font-mono text-slate-100">10.99 dB [Attention]</span>
            </div>
            <div class="flex justify-between items-center py-1">
              <span class="text-slate-300">Continuous CPPS:</span>
              <span id="repCPPS" class="font-mono text-slate-100">34.93 dB [Pass]</span>
            </div>
          </div>

          <div class="bg-slate-900/60 p-4 rounded-lg border border-slate-700/40 space-y-2">
            <div class="text-xs font-semibold text-slate-400 uppercase">Vocal Fold Source & Resonances</div>
            <div class="flex justify-between items-center py-1 border-b border-slate-800">
              <span class="text-slate-300">Glottal Excitation NAQ:</span>
              <span id="repNAQ" class="font-mono text-slate-100">0.092 (Modal Closure)</span>
            </div>
            <div class="flex justify-between items-center py-1 border-b border-slate-800">
              <span class="text-slate-300">Vocal Tract Length:</span>
              <span id="repVTL" class="font-mono text-slate-100">17.2 cm (Adult Male)</span>
            </div>
            <div class="flex justify-between items-center py-1 border-b border-slate-800">
              <span class="text-slate-300">Prosodic Inflection:</span>
              <span id="repProsody" class="font-mono text-slate-100">Expressive / Dynamic</span>
            </div>
            <div class="flex justify-between items-center py-1">
              <span class="text-slate-300">Phonetic Articulation:</span>
              <span id="repVSA" class="font-mono text-slate-100">VSA: 111,332 Hz²</span>
            </div>
          </div>
        </div>

        <!-- Raw Text Report Output -->
        <div class="mt-4">
          <div class="flex items-center justify-between mb-2">
            <span class="text-xs text-slate-400 font-semibold uppercase">Diagnostic Narrative</span>
            <button onclick="copyReportText()" class="text-xs text-sky-400 hover:text-sky-300 font-medium">Copy Text</button>
          </div>
          <pre id="rawReportText" class="bg-slate-950 p-4 rounded-lg font-mono text-xs text-emerald-400 overflow-x-auto border border-slate-800 leading-relaxed"></pre>
        </div>
      </div>
    </div>

    <!-- Tab 2: 6-Panel High-Res Visual Dashboard -->
    <div id="tabContent-visual" class="hidden space-y-4">
      <div class="bg-slate-800/80 border border-slate-700/60 rounded-xl p-5 flex flex-col items-center">
        <div class="w-full flex justify-between items-center mb-4">
          <div>
            <h3 class="text-base font-bold text-white">6-Panel Acoustic & Biometric Multi-Plot</h3>
            <p class="text-xs text-slate-400">Waveform & VAD | STFT Spectrogram & Formants | Pitch Contour | LPC Envelope | MFCCs | Vowel Space</p>
          </div>
          <a id="btnDownloadImg" href="/api/image?file=voice_dashboard.png" download="voice_dashboard.png" class="bg-sky-500 hover:bg-sky-600 text-white text-xs font-semibold px-4 py-2 rounded-lg transition flex items-center gap-1.5">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path></svg>
            Download High-Res PNG
          </a>
        </div>
        <div class="w-full border border-slate-700/80 rounded-lg overflow-hidden bg-slate-950 flex justify-center p-2">
          <img id="dashboardImg" src="/api/image?file=voice_dashboard.png" alt="6-Panel Diagnostic Dashboard" class="max-w-full h-auto rounded shadow-lg">
        </div>
      </div>
    </div>

    <!-- Tab 3: Tabular Dataset Explorer -->
    <div id="tabContent-dataset" class="hidden space-y-4">
      <div class="bg-slate-800/80 border border-slate-700/60 rounded-xl p-5">
        <div class="flex justify-between items-center mb-4">
          <h3 class="text-base font-bold text-white">voice_dataset.csv Records</h3>
          <a href="/voice_dataset.csv" download class="bg-slate-700 hover:bg-slate-600 text-slate-200 text-xs px-3 py-1.5 rounded-lg transition">Download Raw CSV</a>
        </div>
        <div class="overflow-x-auto rounded-lg border border-slate-700/60">
          <table class="min-w-full text-xs text-left text-slate-300">
            <thead class="bg-slate-900 text-slate-400 uppercase font-semibold border-b border-slate-700">
              <tr>
                <th class="px-3 py-2">File</th>
                <th class="px-3 py-2">Demographic</th>
                <th class="px-3 py-2">VTL (cm)</th>
                <th class="px-3 py-2">Pitch (Hz)</th>
                <th class="px-3 py-2">F1 (Hz)</th>
                <th class="px-3 py-2">F2 (Hz)</th>
                <th class="px-3 py-2">Jitter (%)</th>
                <th class="px-3 py-2">HNR (dB)</th>
                <th class="px-3 py-2">CPPS (dB)</th>
                <th class="px-3 py-2">Clinical Assessment</th>
              </tr>
            </thead>
            <tbody id="datasetTableBody" class="divide-y divide-slate-800">
            </tbody>
          </table>
        </div>
      </div>
    </div>

  </main>

  <footer class="border-t border-slate-800 py-4 text-center text-xs text-slate-500">
    vox-dsp Audio Processing Pipeline &copy; 2026. Powered by SciPy, Librosa & Praat.
  </footer>

  <script>
    let currentProfile = null;

    async function loadInitialData() {
      // 1. Load available files
      try {
        const res = await fetch('/api/files');
        const files = await res.json();
        const sel = document.getElementById('fileSelect');
        sel.innerHTML = '';
        files.forEach(f => {
          const opt = document.createElement('option');
          opt.value = f;
          opt.textContent = f;
          sel.appendChild(opt);
        });
      } catch (e) {
        console.error(e);
      }

      // 2. Set audio player source
      updateActiveAudio();

      // 3. Run initial analysis on active file
      triggerAnalyze();

      // 4. Load dataset records
      loadDatasetTable();
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
      // 1. Metric Cards
      const demo = data.demographics || {};
      const pitch = data.pitch || {};
      const formants = data.formants || {};
      const clinical = data.clinical_vowel_biometrics || {};
      const cont = data.continuous_speech_biometrics || {};
      const diag = data.diagnostic_evaluation || {};

      document.getElementById('cardSpeaker').textContent = demo.demographic_class || 'Adult Male';
      document.getElementById('cardVTL').textContent = `${demo.vocal_tract_length_cm || 17.2} cm`;
      document.getElementById('cardLPCOrder').textContent = demo.adapted_lpc_order || 18;

      document.getElementById('cardF0').textContent = pitch.median_f0 || 0;
      document.getElementById('cardSpread').textContent = `${pitch.pitch_range_semitones || 0} semitones`;
      document.getElementById('cardDuty').textContent = `${((pitch.voiced_ratio || 0) * 100).toFixed(1)}% voiced`;

      document.getElementById('cardFormants').textContent = `F1: ${Math.round(formants.f1_hz || 0)} | F2: ${Math.round(formants.f2_hz || 0)}`;
      document.getElementById('cardVowel').textContent = formants.dominant_vowel || 'N/A';
      document.getElementById('cardVSA').textContent = `${(formants.vowel_space_area_hz2 || 0).toLocaleString()} Hz²`;

      document.getElementById('cardJitter').textContent = `${clinical.jitter_local_percent || 0}%`;
      document.getElementById('cardShimmer').textContent = `${clinical.shimmer_local_percent || 0}%`;
      document.getElementById('cardHNR').textContent = `${clinical.hnr_db || 0} dB`;
      document.getElementById('cardCPPS').textContent = `${cont.cpps_db || 0} dB`;

      // Badge
      const badge = document.getElementById('cardOverallBadge');
      badge.textContent = diag.jitter_flag || 'Pass';
      badge.className = 'text-xs px-2 py-0.5 rounded font-medium ' + 
        (diag.jitter_flag === 'Pass' ? 'badge-pass' : diag.jitter_flag === 'Borderline' ? 'badge-borderline' : 'badge-attention');

      // 2. Report Tab
      document.getElementById('repJitter').textContent = `${clinical.jitter_local_percent}% [${diag.jitter_flag}]`;
      document.getElementById('repShimmer').textContent = `${clinical.shimmer_local_percent}% [${diag.shimmer_flag}]`;
      document.getElementById('repHNR').textContent = `${clinical.hnr_db} dB [${diag.hnr_flag}]`;
      document.getElementById('repCPPS').textContent = `${cont.cpps_db} dB [${diag.cpps_flag}]`;

      document.getElementById('repNAQ').textContent = `${cont.normalized_amplitude_quotient || 0.09} (Glottal Closure)`;
      document.getElementById('repVTL').textContent = `${demo.vocal_tract_length_cm} cm (${demo.demographic_class})`;
      document.getElementById('repProsody').textContent = diag.intonation_style || 'Normal';
      document.getElementById('repVSA').textContent = `Area: ${(formants.vowel_space_area_hz2 || 0).toLocaleString()} Hz²`;

      document.getElementById('rawReportText').textContent = data.report_text || 'Report generated successfully.';
    }

    async function renderDashboardImg() {
      const file = document.getElementById('fileSelect').value || 'voice.wav';
      const img = document.getElementById('dashboardImg');
      img.style.opacity = '0.5';
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
          tr.className = 'hover:bg-slate-800/50 transition';
          tr.innerHTML = `
            <td class="px-3 py-2 font-mono text-sky-400">${r.File || ''}</td>
            <td class="px-3 py-2">${r.Demographic || ''}</td>
            <td class="px-3 py-2">${r.Est_VTL_cm || ''}</td>
            <td class="px-3 py-2 font-mono">${r.Pitch_Median_Hz || ''}</td>
            <td class="px-3 py-2 font-mono">${r.Formant_F1_Hz || ''}</td>
            <td class="px-3 py-2 font-mono">${r.Formant_F2_Hz || ''}</td>
            <td class="px-3 py-2 font-mono">${r.Jitter_Local_Pct || ''}</td>
            <td class="px-3 py-2 font-mono">${r.HNR_dB || ''}</td>
            <td class="px-3 py-2 font-mono">${r.CPPS_dB || ''}</td>
            <td class="px-3 py-2 truncate max-w-xs text-slate-400">${r.Clinical_Assessment || ''}</td>
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
        document.getElementById(`tabBtn-${t}`).className = 'py-3 px-1 text-slate-400 hover:text-slate-200 flex items-center gap-2';
      });
      document.getElementById(`tabContent-${tabId}`).classList.remove('hidden');
      document.getElementById(`tabBtn-${tabId}`).className = 'py-3 px-1 tab-active flex items-center gap-2';
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
      alert('Diagnostic report copied to clipboard!');
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

            # Update CSV
            csv_path = os.path.join(WORKSPACE_DIR, "voice_dataset.csv")
            save_rows_to_csv([flat], csv_path)

            self.send_json(profile)
            return

        # 5. Render/Get 6-Panel Dashboard image
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

    def log_message(self, format, *args):
        # Suppress noisy HTTP request logging in terminal
        return


def start_server(port=PORT, open_browser=True):
    server = HTTPServer(("127.0.0.1", port), DashboardHTTPRequestHandler)
    url = f"http://localhost:{port}"
    print("=" * 65)
    print(f"  vox-dsp Centralized Access Dashboard is running!")
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
    parser = argparse.ArgumentParser(description="vox-dsp Centralized Access Dashboard Server")
    parser.add_argument("--port", type=int, default=PORT, help="Port to listen on (default: 8050)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    args = parser.parse_args()

    start_server(port=args.port, open_browser=not args.no_browser)
