import { Linkedin, Github, Mail, BookOpen, Sparkles, Brain } from 'lucide-react';

export const Overview = () => {
  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 py-6 bg-background">
      {/* Hero Section */}
      <div className="flex flex-col md:flex-row items-center md:items-start gap-6 mb-8 mt-2">
        {/* Profile Picture */}
        <div className="relative group">
          <div className="absolute -inset-1 bg-gradient-to-r from-brand-orange to-brand-blue rounded-2xl blur opacity-25 group-hover:opacity-50 transition duration-1000 group-hover:duration-200"></div>
          <div className="relative w-32 h-32 rounded-2xl overflow-hidden border border-white/10 dark:border-black/50 shadow-2xl flex-shrink-0 bg-background">
            <img
              src="/images/profile.jpg"
              alt="Zizhao Hu"
              className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
              onError={(e) => {
                const target = e.target as HTMLImageElement;
                target.style.display = 'none';
                const parent = target.parentElement;
                if (parent) {
                  parent.innerHTML = `
                    <div class="w-full h-full bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center text-white text-3xl font-bold">
                      ZH
                    </div>
                  `;
                }
              }}
            />
          </div>
        </div>

        {/* Main Info */}
        <div className="flex-1 text-center md:text-left">
          <div className="flex flex-col md:flex-row md:items-center gap-2 md:gap-4 mb-2">
            <h1 className="text-3xl font-bold tracking-tight text-foreground font-heading">
              Zizhao Hu
            </h1>
            <div className="flex items-center justify-center md:justify-start gap-1.5">
              <span className="flex h-2 w-2 rounded-full bg-green-500"></span>
              <span className="text-[10px] font-medium uppercase tracking-wider text-green-600 dark:text-green-400">Available for Collaboration</span>
            </div>
          </div>

          <p className="text-lg text-brand-orange font-medium mb-3 font-heading">
            AI Researcher · PhD Student at USC · MOVE Fellow
          </p>

          <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed mb-4 max-w-2xl">
            I build AI systems that improve themselves while remaining under control. My work spans
            <span className="text-gray-900 dark:text-white font-semibold"> multi-agent systems & synthetic data</span>,
            <span className="text-gray-900 dark:text-white font-semibold"> brain-inspired neural architectures</span>, and
            <span className="text-gray-900 dark:text-white font-semibold"> continual learning</span>.
            Former MOVE Fellow at Handshake AI, specializing in frontier model training and safety.
          </p>

          {/* Quick Stats */}
          <div className="flex flex-wrap justify-center md:justify-start gap-2">
            <div className="px-3 py-1 rounded-full bg-gray-100 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 text-gray-700 dark:text-gray-300 text-[10px] font-semibold uppercase tracking-tight">
              🎓 USC PhD Student
            </div>
            <div className="px-3 py-1 rounded-full bg-gray-100 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 text-gray-700 dark:text-gray-300 text-[10px] font-semibold uppercase tracking-tight">
              🤖 GLAMOUR / MINDS Lab
            </div>
            <div className="px-3 py-1 rounded-full bg-blue-50 dark:bg-blue-900/20 border border-blue-100 dark:border-blue-900/50 text-blue-700 dark:text-blue-400 text-[10px] font-semibold uppercase tracking-tight">
              🌐 Handshake AI (Alumni)
            </div>
          </div>
        </div>
      </div>

      {/* Highlights */}
      <div className="grid md:grid-cols-3 gap-3 mb-6">
        <div className="p-3 rounded-lg bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 border border-blue-100 dark:border-blue-800">
          <h3 className="text-sm font-semibold text-foreground mb-1 flex items-center gap-1.5 font-heading">
            <Brain className="w-3.5 h-3.5 text-blue-600" />
            Research Focus
          </h3>
          <p className="text-xs text-gray-600 dark:text-gray-400">
            Multi-agent systems, self-improving AI via synthetic data, brain-inspired architectures, continual learning
          </p>
        </div>
        <div className="p-3 rounded-lg bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 border border-purple-100 dark:border-purple-800">
          <h3 className="text-sm font-semibold text-foreground mb-1 flex items-center gap-1.5 font-heading">
            <BookOpen className="w-3.5 h-3.5 text-purple-600" />
            Featured Work
          </h3>
          <p className="text-xs text-gray-600 dark:text-gray-400">
            Published at ICMI, NeurIPS workshops. Reviewer for ICLR, ICML, NeurIPS.
          </p>
        </div>
        <div className="p-3 rounded-lg bg-gradient-to-br from-orange-50 to-yellow-50 dark:from-orange-900/20 dark:to-yellow-900/20 border border-orange-100 dark:border-orange-800">
          <h3 className="text-sm font-semibold text-foreground mb-1 flex items-center gap-1.5 font-heading">
            <Sparkles className="w-3.5 h-3.5 text-orange-600" />
            Beyond
          </h3>
          <p className="text-xs text-gray-600 dark:text-gray-400">
            Spade A K-pop dancer at USC, fitness enthusiast, Asian food connoisseur
          </p>
        </div>
      </div>

      {/* Links */}
      <div className="flex items-center justify-center gap-2 flex-wrap">
        <a
          href="https://scholar.google.com/citations?user=A8J42tQAAAAJ"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-brand-orange hover:opacity-90 text-white transition-all text-xs font-medium font-heading"
        >
          <BookOpen className="w-3.5 h-3.5" />
          Google Scholar
        </a>
        <a
          href="https://linkedin.com/in/zizhao-hu"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300 transition-colors text-xs font-medium"
        >
          <Linkedin className="w-3.5 h-3.5" />
          LinkedIn
        </a>
        <a
          href="https://github.com/zizhao-hu"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300 transition-colors text-xs font-medium"
        >
          <Github className="w-3.5 h-3.5" />
          GitHub
        </a>
        <a
          href="mailto:zizhaoh@usc.edu"
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300 transition-colors text-xs font-medium"
        >
          <Mail className="w-3.5 h-3.5" />
          Contact
        </a>
      </div>
    </div>
  );
};
