/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        warning: {
          DEFAULT: "#b45309",
          bg: "#fffbeb",
          border: "#fcd34d",
        },
      },
    },
  },
  plugins: [],
};
