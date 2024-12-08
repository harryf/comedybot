Jekyll::Hooks.register :site, :after_init do |site|
  # Get all subdirectories in assets/audio
  audio_dir = File.join(site.source, 'assets', 'audio')
  next unless File.directory?(audio_dir)

  Dir.entries(audio_dir).each do |entry|
    next if entry.start_with?('.') # Skip hidden files/directories
    folder_path = File.join(audio_dir, entry)
    next unless File.directory?(folder_path)

    # Create a new page for this folder
    player_file = File.join(site.source, '_players', "#{entry}.md")
    
    unless File.exist?(player_file)
      content = <<~CONTENT
        ---
        title: "#{entry.gsub('_', ' ')}"
        transcript_path: "#{entry}"
        ---
      CONTENT

      File.write(player_file, content)
    end
  end
end
