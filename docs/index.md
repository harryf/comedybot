---
layout: default
title: Comedy Bot
---

<div class="home">
  <div class="welcome">
    <h1>{{ site.title }}</h1>
    <p class="page-description">
      {{ site.description }}
    </p>
  </div>

  <div class="section-links">
    <div class="section-card">
      <h2><a href="{{ site.baseurl }}/bits/">Comedy Bits</a></h2>
      <p>View all comedy bits chronologically, with performance dates and venues.</p>
    </div>
    
    <div class="section-card">
      <h2><a href="{{ site.baseurl }}/themes/">Themes</a></h2>
      <p>Browse bits by theme, such as Cultural Differences, Family Life, and more.</p>
    </div>

    <div class="section-card">
      <h2><a href="{{ site.baseurl }}/joke-types/">Joke Types</a></h2>
      <p>Explore different styles of comedy, from Wordplay to Observational humor.</p>
    </div>
  </div>

  <div class="recent-shows">
    <h2>Recent Shows</h2>
    <div class="show-list">
      {% for show in site.data.shows %}
        <div class="show-card">
          <div>
            <h3>{{ show.name_of_show }}</h3>
            <ul class="show-details">
              <li class="comedian">
                <i class="fas fa-calendar-alt"></i>
                {{ show.date_of_show }} - {{ show.comedian }}
              </li>
              <li class="venue">
                <i class="fas fa-map-marker-alt"></i>
                {% if show.link_to_venue_on_google_maps %}
                  <a href="{{ show.link_to_venue_on_google_maps }}" target="_blank" class="venue-link">
                    {{ show.name_of_venue }}
                  </a>
                {% else %}
                  {{ show.name_of_venue }}
                {% endif %}
              </li>
              {% if show.notes %}
                <li class="notes">
                  <i class="fas fa-info-circle"></i>
                  {{ show.notes }}
                </li>
              {% endif %}
            </ul>
          </div>
          <a href="{{ site.baseurl }}/player/{{ show.folder }}" class="listen-button mt-4">
            <i class="fas fa-headphones"></i>
            Listen to Show
          </a>
        </div>
      {% endfor %}
    </div>
  </div>
</div>

<style>
.welcome {
  text-align: center;
  margin: 2rem 0;
  padding: 2rem;
  background: #f8f9fa;
  border-radius: 8px;
}

.welcome h1 {
  color: #333;
  margin-bottom: 1rem;
}

.page-description {
  color: #666;
  font-size: 1.1rem;
  max-width: 800px;
  margin: 0 auto;
}

.section-links {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 2rem;
  margin: 2rem 0;
  padding: 0 1rem;
}

.section-card {
  background: white;
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 1.5rem;
  text-align: center;
  transition: transform 0.2s, box-shadow 0.2s;
  height: 200px;
  display: flex;
  flex-direction: column;
  justify-content: center;
}

.section-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
}

.section-card h2 {
  color: #333;
  margin-bottom: 1rem;
}

.section-card h2 a {
  color: inherit;
  text-decoration: none;
}

.section-card p {
  color: #666;
  margin-bottom: 0;
}

.recent-shows {
  margin-top: 3rem;
  padding: 0 1rem;
}

.recent-shows h2 {
  text-align: center;
  margin-bottom: 2rem;
  color: #333;
}

.show-list {
  display: grid;
  gap: 2rem;
  grid-template-columns: repeat(3, 1fr);
}

.show-card {
  background: white;
  border: 1px solid #ddd;
  border-radius: 8px;
  overflow: hidden;
  transition: transform 0.2s, box-shadow 0.2s;
}

.show-card:hover {
  transform: translateY(-3px);
  box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
}

.show-link {
  color: inherit;
  text-decoration: none;
  padding: 1.5rem;
  display: block;
}

.show-link h3 {
  color: #333;
  font-size: 1.2rem;
  margin: 0 0 1rem;
}

.show-details {
  list-style: none;
  padding: 0;
  margin: 0;
}

.show-details li {
  margin: 0.5rem 0;
  font-size: 0.95rem;
  line-height: 1.4;
  color: #666;
}

.show-date {
  color: #e74c3c !important;
  font-weight: 600;
}

.comedian {
  color: #34495e !important;
}

.venue-link {
  color: #3498db;
  text-decoration: none;
}

.venue-link:hover {
  text-decoration: underline;
}

.notes {
  color: #95a5a6 !important;
  font-style: italic;
}

.listen-button {
  background-color: #3498db;
  color: #fff;
  padding: 0.5rem 1rem;
  border: none;
  border-radius: 5px;
  cursor: pointer;
}

.listen-button:hover {
  background-color: #2e6da4;
}

@media (max-width: 992px) {
  .show-list {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (max-width: 768px) {
  .section-links {
    grid-template-columns: 1fr;
  }
  
  .show-list {
    grid-template-columns: 1fr;
  }
}
</style>

<script>
document.addEventListener('DOMContentLoaded', function() {
  // Sort show cards by date
  const container = document.querySelector('.show-list');
  const cards = Array.from(container.children);
  
  cards.sort((a, b) => {
    const dateA = new Date(a.dataset.date.replace(',', ''));
    const dateB = new Date(b.dataset.date.replace(',', ''));
    return dateB - dateA;
  });
  
  cards.forEach(card => container.appendChild(card));
});
</script>
