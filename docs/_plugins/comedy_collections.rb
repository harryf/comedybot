module ComedyCollections
  class Generator < Jekyll::Generator
    def generate(site)
      # Load data files
      load_data_files(site)
      
      # Create collections data
      @themes_collection = process_collection(site.data['themes'] || {}, site.data['canonical_bits'] || {}, site.data['bits'] || {})
      @joke_types_collection = process_collection(site.data['joke_types'] || {}, site.data['canonical_bits'] || {}, site.data['bits'] || {})
      @bits_collection = process_bits(site.data['canonical_bits'] || {}, site.data['bits'] || {})
      
      # Add to site
      site.data['themes_collection'] = @themes_collection
      site.data['joke_types_collection'] = @joke_types_collection
      site.data['bits_collection'] = @bits_collection
    end
    
    private
    
    def load_data_files(site)
      # Load all bit files
      site.data['bits'] = {}
      Dir.glob(File.join(site.source, '_data', 'bits', '*.json')).each do |file|
        bit_id = File.basename(file, '.json')
        site.data['bits'][bit_id] = JSON.parse(File.read(file))
      end
      
      # Load canonical bits
      canonical_bits_file = File.join(site.source, '_data', 'canonical_bits.json')
      if File.exist?(canonical_bits_file)
        site.data['canonical_bits'] = JSON.parse(File.read(canonical_bits_file))
      else
        site.data['canonical_bits'] = {}
      end
      
      # Load themes
      themes_file = File.join(site.source, '_data', 'themes.json')
      if File.exist?(themes_file)
        site.data['themes'] = JSON.parse(File.read(themes_file))
      else
        site.data['themes'] = {}
      end
      
      # Load joke types
      joke_types_file = File.join(site.source, '_data', 'joke_types.json')
      if File.exist?(joke_types_file)
        site.data['joke_types'] = JSON.parse(File.read(joke_types_file))
      else
        site.data['joke_types'] = {}
      end
      
      # Create a reverse lookup for canonical names
      site.data['bit_to_canonical'] = {}
      site.data['canonical_bits'].each do |canonical_name, bit_ids|
        bit_ids.each do |bit_id|
          site.data['bit_to_canonical'][bit_id] = canonical_name
        end
      end
    end
    
    def process_collection(collection_data, canonical_bits, bits_data)
      result = {}
      
      collection_data.each do |category, bit_ids|
        result[category] = bit_ids.map do |bit_id|
          next unless bits_data[bit_id]
          bit_data = bits_data[bit_id]
          canonical_name = canonical_bits.find { |name, ids| ids.include?(bit_id) }&.first
          
          {
            'bit_id' => bit_id,
            'canonical_name' => canonical_name,
            'date_of_show' => bit_data['show_info']['date_of_show'],
            'lpm' => bit_data['show_info']['lpm'],
            'venue' => bit_data['show_info']['name_of_venue']
          }
        end.compact
      end
      
      result
    end
    
    def process_bits(canonical_bits, bits_data)
      result = {}
      
      canonical_bits.each do |canonical_name, bit_ids|
        result[canonical_name] = bit_ids.map do |bit_id|
          next unless bits_data[bit_id]
          bit_data = bits_data[bit_id]
          
          {
            'bit_id' => bit_id,
            'canonical_name' => canonical_name,
            'date_of_show' => bit_data['show_info']['date_of_show'],
            'lpm' => bit_data['show_info']['lpm'],
            'venue' => bit_data['show_info']['name_of_venue'],
            'start' => bit_data['bit_info']['start'],
            'show_identifier' => bit_data['show_info']['show_identifier'],
            'transcript' => bit_data['transcript']['lines']
          }
        end.compact.sort_by { |bit| bit['date_of_show'] }.reverse
      end
      
      result
    end
  end
end
