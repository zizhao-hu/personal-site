import { Linkedin, Github, Mail, BookOpen, GraduationCap } from 'lucide-react';

export const Overview = () => {

  return (
    <div className="max-w-4xl mx-auto px-4 py-6 bg-background border-b border-border">
      {/* Personal Introduction */}
      <div className="rounded-xl p-6 flex flex-row items-stretch gap-6 leading-relaxed text-left max-w-4xl mx-auto mb-6 bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-gray-800 dark:to-gray-700">
        {/* Profile Picture */}
        <div className="w-32 rounded-2xl overflow-hidden border-4 border-blue-200 dark:border-blue-700 shadow-lg flex-shrink-0 flex items-center justify-center">
          <img 
            src="https://zizhao-hu.github.io/assets/img/prof_pic.jpg" 
            alt="Zizhao Hu" 
            className="w-full h-full object-cover"
            onError={(e) => {
              // Fallback to initials if image fails to load
              const target = e.target as HTMLImageElement;
              target.style.display = 'none';
              const parent = target.parentElement;
              if (parent) {
                parent.innerHTML = `
                  <div class="w-full h-full bg-blue-600 dark:bg-blue-500 flex items-center justify-center text-white text-xl font-bold rounded-2xl">
                    ZH
                  </div>
                `;
              }
            }}
          />
        </div>
        
        {/* Text Content */}
        <div className="flex-1">
          <div className="flex items-center gap-3 mb-2">
            <GraduationCap size={32} className="text-blue-600 dark:text-blue-400"/>
            <h1 className="text-2xl font-bold text-gray-800 dark:text-gray-200">Zizhao Hu</h1>
          </div>
          <p className="text-base text-gray-700 dark:text-gray-300 mb-2 leading-relaxed">
            CS Ph.D. student at USC's Viterbi School of Engineering, affiliated with the GLAMOUR Lab under Professor Jesse Thomason and Professor Mohammad Rostami. 
            My research focuses on synthetic data generation, multi-agent systems, and multi-modal fusion, 
            with expertise in developing self-improving AI agents and distributed learning systems.
          </p>
          <p className="text-sm text-gray-600 dark:text-gray-400 font-medium">
            Former researcher at USC iLab, Georgia Tech's Agile Systems Lab, and Photonics Research Group. 
            Available for consulting, collaborations, and research partnerships.
          </p>
        </div>
      </div>

      {/* Quick Links - With Text Labels */}
      <div className="flex items-center justify-center gap-6">
        <a 
          href="https://scholar.google.com/citations?user=A8J42tQAAAAJ" 
          target="_blank" 
          rel="noopener noreferrer"
          className="flex items-center gap-2 p-3 rounded-lg bg-blue-100 dark:bg-blue-900/30 hover:bg-blue-200 dark:hover:bg-blue-900/50 transition-all duration-200 group"
          title="Google Scholar"
        >
          <BookOpen className="w-4 h-4 text-blue-600 dark:text-blue-400 group-hover:scale-110 transition-transform" />
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Scholar</span>
        </a>
        <a 
          href="https://linkedin.com/in/YOUR_LINKEDIN_USERNAME" 
          target="_blank" 
          rel="noopener noreferrer"
          className="flex items-center gap-2 p-3 rounded-lg bg-blue-100 dark:bg-blue-900/30 hover:bg-blue-200 dark:hover:bg-blue-900/50 transition-all duration-200 group"
          title="LinkedIn"
        >
          <Linkedin className="w-4 h-4 text-blue-600 dark:text-blue-400 group-hover:scale-110 transition-transform" />
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">LinkedIn</span>
        </a>
        <a 
          href="https://github.com/YOUR_GITHUB_USERNAME" 
          target="_blank" 
          rel="noopener noreferrer"
          className="flex items-center gap-2 p-3 rounded-lg bg-gray-100 dark:bg-gray-900/30 hover:bg-gray-200 dark:hover:bg-gray-900/50 transition-all duration-200 group"
          title="GitHub"
        >
          <Github className="w-4 h-4 text-gray-700 dark:text-gray-300 group-hover:scale-110 transition-transform" />
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">GitHub</span>
        </a>
        <a 
          href="mailto:zizhaoh@usc.edu" 
          className="flex items-center gap-2 p-3 rounded-lg bg-red-100 dark:bg-red-900/30 hover:bg-red-200 dark:hover:bg-red-900/50 transition-all duration-200 group"
          title="Email"
        >
          <Mail className="w-4 h-4 text-red-600 dark:text-red-400 group-hover:scale-110 transition-transform" />
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Email</span>
        </a>
      </div>
    </div>
  );
};
