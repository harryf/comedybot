module Jekyll
  class MetadataGenerator < Generator
    def generate(site)
      site.data['shows'] = []
      
      Dir.glob(File.join(site.source, 'assets/audio/*/metadata.json')).each do |file|
        folder = File.basename(File.dirname(file))
        begin
          metadata = JSON.parse(File.read(file))
          metadata['folder'] = folder
          site.data['shows'] << metadata
        rescue => e
          puts "Error reading #{file}: #{e.message}"
        end
      end
      
      site.data['shows'].sort_by! { |show| DateTime.parse(show['date_of_show']) }.reverse!
    end
  end
end
