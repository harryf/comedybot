module Jekyll
  class BitPageGenerator < Generator
    safe true

    def generate(site)
      puts "Starting BitPageGenerator..."
      
      # Load data files explicitly
      load_data_files(site)
      
      # Debug data
      puts "Canonical bits: #{site.data['canonical_bits'] ? site.data['canonical_bits'].keys.join(', ') : 'nil'}"
      puts "Number of bits: #{site.data['bits'] ? site.data['bits'].keys.length : 0}"
      
      # Ensure we have the necessary data
      unless site.data['canonical_bits'] && site.data['bits']
        puts "Missing required data: canonical_bits or bits"
        return
      end

      # Create a page for each bit
      site.data['canonical_bits'].each do |canonical_name, bit_ids|
        puts "Processing canonical name: #{canonical_name}"
        bit_ids.each do |bit_id|
          puts "  Processing bit: #{bit_id}"
          # Skip if we don't have data for this bit
          unless site.data['bits'][bit_id]
            puts "  No data found for bit: #{bit_id}"
            next
          end

          # Create a new page
          page = BitPage.new(site, site.source, 'bits', bit_id, canonical_name)
          puts "  Created page: #{page.url}"
          site.pages << page
        end
      end
      puts "BitPageGenerator finished. Total pages: #{site.pages.count}"
    end
    
    private
    
    def load_data_files(site)
      # Load canonical bits
      canonical_bits_file = File.join(site.source, '_data', 'canonical_bits.json')
      if File.exist?(canonical_bits_file)
        puts "Loading canonical bits from #{canonical_bits_file}"
        site.data['canonical_bits'] = JSON.parse(File.read(canonical_bits_file))
      else
        puts "Warning: canonical_bits.json not found at #{canonical_bits_file}"
      end
      
      # Load all bit files
      site.data['bits'] = {}
      bits_dir = File.join(site.source, '_data', 'bits')
      if Dir.exist?(bits_dir)
        puts "Loading bits from #{bits_dir}"
        Dir.glob(File.join(bits_dir, '*.json')).each do |file|
          bit_id = File.basename(file, '.json')
          site.data['bits'][bit_id] = JSON.parse(File.read(file))
          puts "  Loaded bit: #{bit_id}"
        end
      else
        puts "Warning: bits directory not found at #{bits_dir}"
      end
    end
  end

  class BitPage < Page
    def initialize(site, base, dir, bit_id, canonical_name)
      @site = site
      @base = base
      @dir = dir
      @name = "#{bit_id}.html"

      self.process(@name)
      self.read_yaml(File.join(base, '_layouts'), 'bit.html')
      
      self.data['title'] = canonical_name
      self.data['bit_id'] = bit_id
      self.data['canonical_name'] = canonical_name
      self.data['layout'] = 'bit'
    end
  end
end
