/**
 * Brand logo components — small inline SVGs in each org's brand colors.
 * Sized via parent (typically w-7 h-7 / w-9 h-9). Decorative only.
 */

type LogoProps = { className?: string };

export const GeorgiaTechLogo = ({ className = '' }: LogoProps) => (
  <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" className={className} aria-label="Georgia Tech">
    <rect width="24" height="24" rx="3" fill="#003057" />
    <text
      x="12" y="16.5" textAnchor="middle"
      fill="#B3A369"
      fontFamily="Georgia, 'Times New Roman', serif"
      fontWeight="900" fontSize="11"
      letterSpacing="-0.5"
    >GT</text>
  </svg>
);

export const USCLogo = ({ className = '' }: LogoProps) => (
  <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" className={className} aria-label="USC">
    <rect width="24" height="24" rx="3" fill="#990000" />
    <text
      x="12" y="16" textAnchor="middle"
      fill="#FFCC00"
      fontFamily="Georgia, 'Times New Roman', serif"
      fontWeight="900" fontSize="9"
    >USC</text>
  </svg>
);

// Google Cloud — multi-color G + cloud, simplified.
// Uses Google's 4 brand colors arranged as a stylized G.
export const GoogleCloudLogo = ({ className = '' }: LogoProps) => (
  <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" className={className} aria-label="Google Cloud">
    <rect width="24" height="24" rx="3" fill="#fff" stroke="#e8eaed" />
    {/* G shape */}
    <path d="M12 6.5a5.5 5.5 0 1 0 5.39 6.6h-5.39v-2.2h7.7c.07.36.1.73.1 1.1A7.7 7.7 0 1 1 12 4.3c1.95 0 3.72.73 5.07 1.93l-1.55 1.55A5.48 5.48 0 0 0 12 6.5z" fill="#4285F4" />
    {/* Color accents — small dots evoking 4-color logo */}
    <circle cx="6.5" cy="18" r="1.1" fill="#EA4335" />
    <circle cx="9.5" cy="18" r="1.1" fill="#FBBC04" />
    <circle cx="12.5" cy="18" r="1.1" fill="#34A853" />
    <circle cx="15.5" cy="18" r="1.1" fill="#4285F4" />
  </svg>
);

// Scale AI — minimalist "S" in dark on white.
export const ScaleAILogo = ({ className = '' }: LogoProps) => (
  <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" className={className} aria-label="Scale AI">
    <rect width="24" height="24" rx="3" fill="#0F0F0F" />
    {/* Stylized S */}
    <path
      d="M8.5 8.5h7v2.2H10.7c-.6 0-.95.32-.95.83 0 .43.3.74.78.83l3.13.55c1.6.27 2.5 1.18 2.5 2.62 0 1.7-1.27 2.97-3.18 2.97H8.5v-2.2h4.5c.62 0 .98-.3.98-.84 0-.45-.32-.75-.85-.84l-3.07-.54c-1.62-.28-2.55-1.2-2.55-2.65 0-1.78 1.32-2.93 3.0-2.93Z"
      fill="#fff"
    />
  </svg>
);

// Handshake AI — Handshake brand is a dark navy/violet with their "h" mark.
export const HandshakeAILogo = ({ className = '' }: LogoProps) => (
  <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" className={className} aria-label="Handshake AI">
    <rect width="24" height="24" rx="3" fill="#3A4FE0" />
    {/* Lowercase h */}
    <path
      d="M7.5 6.5v11h2.4v-4.6c0-1.3.74-2.05 1.95-2.05 1.18 0 1.78.65 1.78 1.97v4.68h2.4v-5.2c0-2.36-1.2-3.55-3.27-3.55-1.27 0-2.18.45-2.86 1.32V6.5H7.5z"
      fill="#fff"
    />
  </svg>
);

// Meta — interlocking infinity loop in Meta blue.
export const MetaLogo = ({ className = '' }: LogoProps) => (
  <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" className={className} aria-label="Meta">
    <rect width="24" height="24" rx="3" fill="#fff" stroke="#e8eaed" />
    <path
      d="M4.5 12c0-2.6 1.7-4.5 4.1-4.5 1.7 0 2.9.95 4.05 2.6l.85 1.25.85-1.25c1.15-1.65 2.35-2.6 4.05-2.6 2.4 0 4.1 1.9 4.1 4.5 0 2.55-1.7 4.5-4.1 4.5-1.7 0-2.9-.95-4.05-2.6L13.5 12.7l-.85 1.2c-1.15 1.65-2.35 2.6-4.05 2.6-2.4 0-4.1-1.95-4.1-4.5zm2.4 0c0 1.45.85 2.4 2 2.4.95 0 1.7-.55 2.6-1.85L13 11.45l-1.5-1.1c-.9-1.3-1.65-1.85-2.6-1.85-1.15 0-2 .95-2 2.4zm6.85.55l1.5 1.1c.9 1.3 1.65 1.85 2.6 1.85 1.15 0 2-.95 2-2.4 0-1.45-.85-2.4-2-2.4-.95 0-1.7.55-2.6 1.85L13.75 12.55z"
      fill="#0866FF"
    />
  </svg>
);

// OpenAI — black "blossom" approximation: 6-fold radial pattern.
export const OpenAILogo = ({ className = '' }: LogoProps) => (
  <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" className={className} aria-label="OpenAI">
    <rect width="24" height="24" rx="3" fill="#fff" stroke="#e8eaed" />
    <g stroke="#000" strokeWidth="1.3" strokeLinecap="round" fill="none">
      {/* outer hexagon ring */}
      <path d="M12 5.2 L17.5 8.5 L17.5 15.5 L12 18.8 L6.5 15.5 L6.5 8.5 Z" />
      {/* inner radial spokes forming the blossom */}
      <path d="M12 12 L12 5.2" />
      <path d="M12 12 L17.5 8.5" />
      <path d="M12 12 L17.5 15.5" />
      <path d="M12 12 L12 18.8" />
      <path d="M12 12 L6.5 15.5" />
      <path d="M12 12 L6.5 8.5" />
    </g>
  </svg>
);
