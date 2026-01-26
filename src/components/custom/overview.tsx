import { Linkedin, Github, Mail, BookOpen, Sparkles, Brain } from 'lucide-react';

export const Overview = () => {
  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 py-6 bg-background">
      {/* Hero Section */}
      <div className="flex flex-col md:flex-row items-center md:items-start gap-4 mb-6">
        {/* Profile Picture */}
        <div className="w-28 h-28 rounded-xl overflow-hidden border-2 border-blue-200 dark:border-blue-700 shadow-lg flex-shrink-0">
          <img 
            src="/images/profile.jpg" 
            alt="Zizhao Hu" 
            className="w-full h-full object-cover"
            onError={(e) => {
              const target = e.target as HTMLImageElement;
              target.style.display = 'none';
              const parent = target.parentElement;
              if (parent) {
                parent.innerHTML = `
                  <div class="w-full h-full bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center text-white text-2xl font-bold">
                    ZH
                  </div>
                `;
              }
            }}
          />
        </div>
        
        {/* Main Info */}
        <div className="flex-1 text-center md:text-left">
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-1">
            Zizhao Hu
          </h1>
          <p className="text-base text-blue-600 dark:text-blue-400 font-medium mb-2">
            AI Researcher · PhD Student at USC
          </p>
          <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed mb-3 max-w-2xl">
            Building AI systems that improve themselves while remaining under control. My research focuses on 
            <span className="text-purple-600 dark:text-purple-400 font-medium"> multi-agent collaboration</span>, 
            <span className="text-green-600 dark:text-green-400 font-medium"> vision-language architectures</span>, and 
            <span className="text-orange-600 dark:text-orange-400 font-medium"> continual learning</span>.
            MOVE Fellow at Handshake AI, contributing to frontier model training and AI safety research.
          </p>
          
          {/* Quick Stats */}
          <div className="flex flex-wrap justify-center md:justify-start gap-2 mb-3">
            <div className="px-2 py-0.5 rounded-full bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 text-xs font-medium">
              🎓 USC PhD Student
            </div>
            <div className="px-2 py-0.5 rounded-full bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 text-xs font-medium">
              🤖 MINDS Lab / GLAMOUR Lab
            </div>
            <div className="px-2 py-0.5 rounded-full bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300 text-xs font-medium">
              🤝 MOVE Fellow @ Handshake AI
            </div>
          </div>
        </div>
      </div>

      {/* Highlights */}
      <div className="grid md:grid-cols-3 gap-3 mb-6">
        <div className="p-3 rounded-lg bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 border border-blue-100 dark:border-blue-800">
          <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-1 flex items-center gap-1.5">
            <Brain className="w-3.5 h-3.5 text-blue-600" />
            Research Focus
          </h3>
          <p className="text-xs text-gray-600 dark:text-gray-400">
            Multi-agent systems, self-improving AI, vision-language models, continual learning
          </p>
        </div>
        <div className="p-3 rounded-lg bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 border border-purple-100 dark:border-purple-800">
          <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-1 flex items-center gap-1.5">
            <BookOpen className="w-3.5 h-3.5 text-purple-600" />
            Featured Work
          </h3>
          <p className="text-xs text-gray-600 dark:text-gray-400">
            Published at ICMI, NeurIPS workshops. Reviewer for ICLR, ICML, NeurIPS.
          </p>
        </div>
        <div className="p-3 rounded-lg bg-gradient-to-br from-orange-50 to-yellow-50 dark:from-orange-900/20 dark:to-yellow-900/20 border border-orange-100 dark:border-orange-800">
          <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-1 flex items-center gap-1.5">
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
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-blue-600 hover:bg-blue-700 text-white transition-colors text-xs font-medium"
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
