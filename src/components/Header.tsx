import React from "react";
import { Activity } from "lucide-react";

export default function Header() {
  const handleScroll = (id: string) => {
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: "smooth" });
    }
  };

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-white/80 backdrop-blur-md border-b border-outline-variant">
      <div className="flex justify-between items-center w-full px-6 md:px-12 max-w-7xl mx-auto h-16">
        <div 
          onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })} 
          className="flex items-center gap-2 text-xl font-bold text-primary cursor-pointer select-none group"
        >
          <div className="p-1 rounded-lg bg-primary-fixed group-hover:bg-primary/20 transition-colors">
            <Activity className="w-5 h-5 text-primary" />
          </div>
          <span className="font-heading tracking-tight">PulmoAI Research</span>
        </div>
        <div className="hidden md:flex items-center gap-8">
          <button 
            onClick={() => handleScroll("methodology")} 
            className="text-on-surface-variant font-medium text-sm hover:text-primary transition-colors cursor-pointer"
          >
            Methodology
          </button>
          <button 
            onClick={() => handleScroll("prediction")} 
            className="text-on-surface-variant font-medium text-sm hover:text-primary transition-colors cursor-pointer"
          >
            Diagnostics
          </button>
          <button 
            onClick={() => handleScroll("results-section")} 
            className="text-on-surface-variant font-medium text-sm hover:text-primary transition-colors cursor-pointer"
          >
            Analysis Metrics
          </button>
          <button 
            onClick={() => handleScroll("methodology")} 
            className="text-on-surface-variant font-medium text-sm hover:text-primary transition-colors cursor-pointer"
          >
            Performance
          </button>
          <button 
            onClick={() => handleScroll("prediction")}
            className="bg-primary text-on-primary px-4 py-2 rounded-lg font-semibold text-sm hover:bg-primary-container hover:shadow-sm transition-all active:scale-95 cursor-pointer"
          >
            Documentation
          </button>
        </div>
      </div>
    </nav>
  );
}
