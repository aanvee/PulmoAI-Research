import React from "react";

export default function Footer() {
  return (
    <footer className="bg-surface-container-low border-t border-outline-variant mt-24">
      <div className="w-full py-12 px-6 md:px-12 flex flex-col md:flex-row justify-between items-start md:items-center gap-8 max-w-7xl mx-auto">
        <div className="space-y-4">
          <div className="font-heading font-extrabold text-on-surface text-lg">PulmoAI Research</div>
          <div className="flex flex-wrap gap-4">
            <span className="flex items-center gap-1.5 text-secondary text-xs font-semibold">
              <span className="w-2.5 h-2.5 rounded-full bg-primary inline-block"></span> 
              PyTorch v2.1
            </span>
            <span className="flex items-center gap-1.5 text-secondary text-xs font-semibold">
              <span className="w-2.5 h-2.5 rounded-full bg-primary inline-block"></span> 
              DenseNet-121
            </span>
            <span className="flex items-center gap-1.5 text-secondary text-xs font-semibold">
              <span className="w-2.5 h-2.5 rounded-full bg-primary inline-block"></span> 
              React 19 / Tailwind v4
            </span>
          </div>
        </div>

        <div className="flex flex-col md:items-end gap-3">
          <div className="flex flex-wrap gap-6 text-xs font-bold">
            <a className="text-secondary hover:underline hover:text-primary transition-colors" href="#methodology">Tech Stack</a>
            <a className="text-secondary hover:underline hover:text-primary transition-colors" href="#">Privacy Policy</a>
            <a className="text-secondary hover:underline hover:text-primary transition-colors" href="#">IRB Approval</a>
          </div>
          <p className="text-secondary text-xs md:text-right max-w-lg leading-relaxed">
            © 2024 PulmoAI Research. For investigational research purposes only. Not intended for direct clinical diagnostic decisions without physician oversight.
          </p>
        </div>
      </div>
    </footer>
  );
}
