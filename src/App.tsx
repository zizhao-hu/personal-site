import './App.css'
import { Home } from './pages/home/home'
import { Chat } from './pages/chat/chat'
import { Research } from './pages/research/research'
import { Projects } from './pages/projects/projects'
import { Blogs } from './pages/blogs/blogs'
import { BlogPost } from './pages/blogs/blog-post'
import { Tutorials } from './pages/tutorials/tutorials'
import { TutorialDetail } from './pages/tutorials/tutorial-detail'
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
        <Route path="/projects" element={<Projects />} />
        <Route path="/blogs" element={<Blogs />} />
        <Route path="/blogs/:slug" element={<BlogPost />} />
        <Route path="/tutorials" element={<Tutorials />} />
        <Route path="/tutorials/:slug" element={<TutorialDetail />} />
        <Route path="/chat" element={<Chat />} />
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
