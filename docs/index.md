---
layout: default
title: Comedy Transcripts
---

<div class="home">
  <h1>Comedy Transcripts</h1>
  
  <p class="page-description">
    Welcome to the Comedy Transcripts archive. Browse through comedy bits by theme, joke type, or view all bits chronologically.
  </p>

  <div class="section-links">
    <div class="section-card">
      <h2><a href="{{ '/themes/' | relative_url }}">Themes</a></h2>
      <p>Browse bits by theme, such as Cultural Differences, Family Life, and more.</p>
    </div>

    <div class="section-card">
      <h2><a href="{{ '/joke-types/' | relative_url }}">Joke Types</a></h2>
      <p>Explore different styles of comedy, from Wordplay to Observational humor.</p>
    </div>

    <div class="section-card">
      <h2><a href="{{ '/bits/' | relative_url }}">Comedy Bits</a></h2>
      <p>View all comedy bits chronologically, with performance dates and venues.</p>
    </div>
  </div>

  <div class="recent-shows">
    <h2>Recent Shows</h2>
    <div class="show-list">
      {% for show in site.data.shows %}
        <div class="show-card" data-date="{{ show.date_of_show }}">
          <a href="{{ site.baseurl }}/player/{{ show.folder }}" class="show-link">
            <h3 class="show-title">{{ show.name_of_show }}</h3>
            <div class="show-details">
              <div class="show-info">
                <p class="show-date">{{ show.date_of_show }}</p>
                <p class="comedian">{{ show.comedian }}</p>
                <p class="venue">
                  {% if show.link_to_venue_on_google_maps %}
                    <a href="{{ show.link_to_venue_on_google_maps }}" target="_blank" class="venue-link">
                      {{ show.name_of_venue }} 📍
                    </a>
                  {% else %}
                    {{ show.name_of_venue }}
                  {% endif %}
                </p>
                {% if show.notes %}
                  <p class="notes">{{ show.notes }}</p>
                {% endif %}
              </div>
            </div>
          </a>
        </div>
      {% endfor %}
    </div>
  </div>

  <style>
.transcript-list {
  display: grid;
  gap: 1.5rem;
  padding: 1rem;
}

.show-card {
  background: #ffffff;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  transition: transform 0.2s ease, box-shadow 0.2s ease;
  overflow: hidden;
}

.show-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

.show-link {
  display: block;
  padding: 1.5rem;
  text-decoration: none;
  color: inherit;
}

.show-title {
  margin: 0 0 1rem 0;
  font-size: 1.5rem;
  color: #2c3e50;
}

.show-details {
  display: flex;
  gap: 1rem;
}

.show-info {
  flex: 1;
}

.show-date {
  font-size: 1.1rem;
  color: #e74c3c;
  margin: 0 0 0.5rem 0;
  font-weight: 600;
}

.comedian {
  font-size: 1rem;
  color: #34495e;
  margin: 0 0 0.5rem 0;
}

.venue {
  font-size: 1rem;
  color: #7f8c8d;
  margin: 0 0 0.5rem 0;
}

.venue-link {
  color: #3498db;
  text-decoration: none;
}

.venue-link:hover {
  text-decoration: underline;
}

.notes {
  font-size: 0.9rem;
  color: #95a5a6;
  margin: 0;
  font-style: italic;
}

@media (min-width: 768px) {
  .transcript-list {
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  }
}
</style>

<script>
document.addEventListener('DOMContentLoaded', function() {
  // Sort show cards by date
  const container = document.querySelector('.show-list');
  const cards = Array.from(container.children);
  
  cards.sort((a, b) => {
    const dateA = new Date(a.querySelector('.show-date').textContent.replace(',', ''));
    const dateB = new Date(b.querySelector('.show-date').textContent.replace(',', ''));
    return dateB - dateA;
  });
  
  cards.forEach(card => container.appendChild(card));
});
</script>
