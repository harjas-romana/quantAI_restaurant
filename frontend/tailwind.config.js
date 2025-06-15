/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
    "./public/index.html"
  ],
  theme: {
    extend: {
      opacity: {
        '15': '0.15',
        '35': '0.35',
        '45': '0.45',
        '55': '0.55',
        '65': '0.65',
        '85': '0.85',
      },
      backdropOpacity: {
        '15': '0.15',
        '35': '0.35',
        '45': '0.45',
        '55': '0.55',
        '65': '0.65',
        '85': '0.85',
      },
      colors: {
        'glass-white': 'rgba(255, 255, 255, 0.7)',
        'glass-purple': 'rgba(139, 92, 246, 0.7)',
        'glass-pink': 'rgba(217, 70, 239, 0.7)',
        medical: {
          50: '#f0f9ff',  // Light blue tint
          100: '#e0f2fe', // Soft blue
          200: '#bae6fd', // Sky blue
          300: '#7dd3fc', // Bright blue
          400: '#38bdf8', // Vivid blue
          500: '#0ea5e9', // Primary blue
          600: '#0284c7', // Deep blue
          700: '#0369a1', // Rich blue
          800: '#075985', // Dark blue
          900: '#0c4a6e', // Navy blue
        },
        healing: {
          50: '#f0fdf4',   // Mint tint
          100: '#dcfce7',  // Soft mint
          200: '#bbf7d0',  // Light mint
          300: '#86efac',  // Fresh mint
          400: '#4ade80',  // Bright mint
          500: '#22c55e',  // Primary green
          600: '#16a34a',  // Deep green
          700: '#15803d',  // Rich green
          800: '#166534',  // Forest green
          900: '#14532d',  // Dark green
        },
        accent: {
          50: '#fdf2f8',   // Pink tint
          100: '#fce7f3',  // Soft pink
          200: '#fbcfe8',  // Light pink
          300: '#f9a8d4',  // Medium pink
          400: '#f472b6',  // Bright pink
          500: '#ec4899',  // Primary pink
          600: '#db2777',  // Deep pink
          700: '#be185d',  // Rich pink
          800: '#9d174d',  // Dark pink
          900: '#831843',  // Wine pink
        }
      },
      spacing: {
        '18': '4.5rem',
        '22': '5.5rem',
        '30': '7.5rem',
      },
      animation: {
        'float': 'float 6s ease-in-out infinite',
        'pulse-soft': 'pulse-soft 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'slide-up': 'slide-up 0.5s ease-out',
        'slide-down': 'slide-down 0.5s ease-out',
        'fade-in': 'fade-in 0.3s ease-out',
        'bounce-soft': 'bounce-soft 2s infinite',
        'spin-slow': 'spin 3s linear infinite',
        'heartbeat': 'heartbeat 1.5s ease-in-out infinite',
        'shimmer': 'shimmer 2.5s linear infinite',
        'wave': 'wave 2.5s ease-in-out infinite',
        'gradient': 'gradient 15s ease infinite',
        'scale-in': 'scale-in 0.5s ease-out',
        'scale-out': 'scale-out 0.5s ease-in',
        'slide-in-right': 'slide-in-right 0.5s ease-out',
        'slide-in-left': 'slide-in-left 0.5s ease-out',
        'rotate-scale': 'rotate-scale 1.5s ease-in-out infinite',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-20px)' },
        },
        'pulse-soft': {
          '0%, 100%': { opacity: 1 },
          '50%': { opacity: 0.5 },
        },
        'slide-up': {
          '0%': { transform: 'translateY(20px)', opacity: 0 },
          '100%': { transform: 'translateY(0)', opacity: 1 },
        },
        'slide-down': {
          '0%': { transform: 'translateY(-20px)', opacity: 0 },
          '100%': { transform: 'translateY(0)', opacity: 1 },
        },
        'fade-in': {
          '0%': { opacity: 0 },
          '100%': { opacity: 1 },
        },
        'bounce-soft': {
          '0%, 100%': { transform: 'translateY(-5%)', animationTimingFunction: 'cubic-bezier(0.8, 0, 1, 1)' },
          '50%': { transform: 'translateY(0)', animationTimingFunction: 'cubic-bezier(0, 0, 0.2, 1)' },
        },
        heartbeat: {
          '0%': { transform: 'scale(1)' },
          '14%': { transform: 'scale(1.3)' },
          '28%': { transform: 'scale(1)' },
          '42%': { transform: 'scale(1.3)' },
          '70%': { transform: 'scale(1)' },
        },
        shimmer: {
          '0%': { backgroundPosition: '-1000px 0' },
          '100%': { backgroundPosition: '1000px 0' },
        },
        wave: {
          '0%': { transform: 'rotate(0.0deg)' },
          '10%': { transform: 'rotate(14.0deg)' },
          '20%': { transform: 'rotate(-8.0deg)' },
          '30%': { transform: 'rotate(14.0deg)' },
          '40%': { transform: 'rotate(-4.0deg)' },
          '50%': { transform: 'rotate(10.0deg)' },
          '60%': { transform: 'rotate(0.0deg)' },
          '100%': { transform: 'rotate(0.0deg)' },
        },
        gradient: {
          '0%': { backgroundPosition: '0% 50%' },
          '50%': { backgroundPosition: '100% 50%' },
          '100%': { backgroundPosition: '0% 50%' },
        },
        'scale-in': {
          '0%': { transform: 'scale(0)', opacity: 0 },
          '100%': { transform: 'scale(1)', opacity: 1 },
        },
        'scale-out': {
          '0%': { transform: 'scale(1)', opacity: 1 },
          '100%': { transform: 'scale(0)', opacity: 0 },
        },
        'slide-in-right': {
          '0%': { transform: 'translateX(100%)', opacity: 0 },
          '100%': { transform: 'translateX(0)', opacity: 1 },
        },
        'slide-in-left': {
          '0%': { transform: 'translateX(-100%)', opacity: 0 },
          '100%': { transform: 'translateX(0)', opacity: 1 },
        },
        'rotate-scale': {
          '0%': { transform: 'rotate(0deg) scale(1)' },
          '50%': { transform: 'rotate(180deg) scale(1.2)' },
          '100%': { transform: 'rotate(360deg) scale(1)' },
        },
      },
      boxShadow: {
        'inner-lg': 'inset 0 2px 4px 0 rgb(0 0 0 / 0.05)',
        'soft': '0 2px 15px -3px rgb(0 0 0 / 0.1)',
        'glow': '0 0 15px rgba(14, 165, 233, 0.5)',
        'glow-healing': '0 0 15px rgba(34, 197, 94, 0.5)',
        'glow-accent': '0 0 15px rgba(236, 72, 153, 0.5)',
        'float': '0 10px 30px -10px rgba(0, 0, 0, 0.3)',
        'glass': '0 8px 32px 0 rgba(31, 38, 135, 0.37)',
      },
      backdropBlur: {
        'xs': '2px',
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-conic': 'conic-gradient(var(--tw-gradient-stops))',
        'gradient-glass': 'linear-gradient(120deg, rgba(255,255,255,0.3), rgba(255,255,255,0.1))',
      },
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
  ],
  safelist: [
    'bg-opacity-0',
    'bg-opacity-5',
    'bg-opacity-10',
    'bg-opacity-20',
    'bg-opacity-25',
    'bg-opacity-30',
    'bg-opacity-40',
    'bg-opacity-50',
    'bg-opacity-60',
    'bg-opacity-70',
    'bg-opacity-75',
    'bg-opacity-80',
    'bg-opacity-90',
    'bg-opacity-95',
    'bg-opacity-100',
  ],
} 