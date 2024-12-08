/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        blue: {
          600: '#2563EB',
          100: '#DBEAFE',
        },
        gray: {
          50: '#F9FAFB',
          100: '#F3F4F6',
          400: '#9CA3AF',
          500: '#6B7280',
          600: '#4B5563',
          900: '#111827',
        },
      },
      spacing: {
        '18': '4.5rem',
        '20': '5rem',
      },
    },
  },
  plugins: [],
}