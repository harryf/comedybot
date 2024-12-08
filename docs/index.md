---
layout: default
title: Comedy Transcripts
---

<div class="home">
  <h1>Comedy Transcripts</h1>
  
  {% assign audio_dirs = site.static_files | where_exp: "file", "file.path contains '/assets/audio/'" | map: "path" | uniq %}
  {% assign folder_names = "" | split: "" %}
  
  {% for path in audio_dirs %}
    {% assign folder = path | split: "/" | slice: -2, 1 | first %}
    {% unless folder contains "." %}
      {% assign folder_names = folder_names | push: folder %}
    {% endunless %}
  {% endfor %}
  
  {% assign sorted_folders = folder_names | sort | reverse | uniq %}
  
  <ul class="transcript-list">
    {% for folder in sorted_folders %}
      <li>
        <a href="{{ site.baseurl }}/player/{{ folder }}" class="transcript-link" data-folder="{{ folder }}">
          {% assign date = folder | slice: 0, 8 %}
          {% assign title = folder | slice: 9, 100 | replace: "_", " " %}
          {{ date | slice: 0, 4 }}-{{ date | slice: 4, 2 }}-{{ date | slice: 6, 2 }} - {{ title }}
        </a>
      </li>
    {% endfor %}
  </ul>
</div>

<style>
.transcript-list {
  list-style: none;
  padding: 0;
}

.transcript-list li {
  margin: 1em 0;
}

.transcript-link {
  display: block;
  padding: 1em;
  background: #f5f5f5;
  border-radius: 4px;
  text-decoration: none;
  color: #333;
  transition: background-color 0.2s ease;
}

.transcript-link:hover {
  background: #e5e5e5;
  text-decoration: none;
}
</style>
