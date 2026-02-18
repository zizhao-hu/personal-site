import './App.css'
import { Home } from './pages/home/home'
import { Chat } from './pages/chat/chat'
import { Research } from './pages/research/research'
import { LlmVlmResearch } from './pages/research/llm-vlm'
import { ArchitectureResearch } from './pages/research/architecture'
import { ContinualLearningResearch } from './pages/research/continual-learning'
import { SyntheticDataResearch } from './pages/research/synthetic-data'
import { Projects } from './pages/projects/projects'
import { Blogs } from './pages/blogs/blogs'
import { BlogPost } from './pages/blogs/blog-post'
import { Tutorials } from './pages/tutorials/tutorials'
import { TutorialDetail } from './pages/tutorials/tutorial-detail'
import { Tools } from './pages/tools/tools'
import { PipelineDesigner } from './pages/tools/pipeline-designer'
import { FloatingChat } from './components/custom/floating-chat'
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import { ThemeProvider } from './context/ThemeContext'

function AppContent() {
  const location = useLocation();
  const showFloatingChat = location.pathname !== '/chat';

  return (
    <div className="w-full h-screen bg-white dark:bg-gray-900 text-gray-900 dark:text-white">
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/research" element={<Research />} />
        <Route path="/research/llm-vlm" element={<LlmVlmResearch />} />
        <Route path="/research/architecture" element={<ArchitectureResearch />} />
        <Route path="/research/continual-learning" element={<ContinualLearningResearch />} />
        <Route path="/research/synthetic-data" element={<SyntheticDataResearch />} />
        <Route path="/projects" element={<Projects />} />
        <Route path="/blogs" element={<Blogs />} />
        <Route path="/blogs/:slug" element={<BlogPost />} />
        <Route path="/tutorials" element={<Tutorials />} />
        <Route path="/tutorials/:slug" element={<TutorialDetail />} />
        <Route path="/chat" element={<Chat />} />
        <Route path="/tools" element={<Tools />} />
        <Route path="/tools/pipeline-designer" element={<PipelineDesigner />} />
      </Routes>
      {showFloatingChat && <FloatingChat />}
    </div>
  );
}

function App() {
  return (
    <ThemeProvider>
      <Router>
        <AppContent />
      </Router>
    </ThemeProvider>
  )
}

export default App;

