module Jekyll
  class PlayerPage < Page
    def initialize(site, base, dir, name, transcript_path)
      @site = site
      @base = base
      @dir = dir
      @name = "#{name}.html"

      self.process(@name)
      self.read_yaml(File.join(base, '_layouts'), 'player.html')
      
      self.data['title'] = name.gsub('_', ' ')
      self.data['transcript_path'] = transcript_path
      self.data['layout'] = 'player'
    end
  end

  class PlayerPageGenerator < Generator
    safe true

    def generate(site)
      puts "Starting PlayerPageGenerator..."
      
      # Get all subdirectories in assets/audio
      audio_dir = File.join(site.source, 'assets', 'audio')
      unless File.directory?(audio_dir)
        puts "Warning: audio directory not found at #{audio_dir}"
        return
      end

      Dir.entries(audio_dir).each do |entry|
        next if entry.start_with?('.') # Skip hidden files/directories
        folder_path = File.join(audio_dir, entry)
        next unless File.directory?(folder_path)

        puts "Generating player page for: #{entry}"
        
        # Create a new page for this folder
        page = PlayerPage.new(site, site.source, 'player', entry, entry)
        site.pages << page
      end
      
      puts "PlayerPageGenerator finished. Total pages: #{site.pages.count}"
    end
  end
end
