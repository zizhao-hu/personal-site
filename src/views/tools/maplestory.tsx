'use client';

const EMBED_URL = '/games/maplestory/index.html';

export const Maplestory = () => {
  return (
    <div className="fixed inset-0 bg-black">
      <iframe
        src={EMBED_URL}
        title="Ludibrium · MapleStory"
        className="block w-full h-full border-0"
        allow="fullscreen; autoplay; clipboard-read; clipboard-write"
        referrerPolicy="no-referrer-when-downgrade"
      />
    </div>
  );
};
