\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{bachelor2025eng}

% --- Base Class and Default Options ---
\def\@baseclass{report}
\PassOptionsToClass{a4paper,twoside,openright,11pt}{\@baseclass}
\ProcessOptions

% --- Load Base Class ---
\LoadClass{\@baseclass}

% --- Page Layout Configuration ---
% Adjust text width and margins for A4 paper
\textwidth\paperwidth
\advance\textwidth -55mm
\oddsidemargin-1in
\advance\oddsidemargin 30mm
\evensidemargin-1in
\advance\evensidemargin 25mm

% Adjust text height and top margin
\topmargin -1in
\advance\topmargin 2cm
\setlength\textheight{48\baselineskip}
\addtolength\textheight{\topskip}

% Set margin paragraph width
\marginparwidth15mm

% =========================================================================
% Title Page Definition (\maketitle)
% =========================================================================
\renewcommand\maketitle{%
    \begin{titlepage}%
        \let\footnotesize\small
        \let\footnoterule\relax
        \let \footnote \thanks
        \begin{center}%
            % University and Faculty Information
            {\LARGE\textbf{University of Warsaw\\ Faculty of Physics}\par}
            \vspace{1cm plus 1fill} 
    
            % Author Information
            {\Large\@author\par}
            \vspace{0.2cm}
            {\large Record book number: \@studentID\par}
            \vspace{8mm plus .1fill}
    
            % Thesis Title
            {\Huge\textbf{\@title}\par}
            \vspace{8mm plus .1fill}
            {\large 
                Bachelor's thesis\\[3pt]
                in the field of 
                \@fieldOfStudy
            \par}
            \vspace{2cm plus 1.5fill}
    
            % Supervisor Information
            \begin{flushright}\large
                \begin{tabular}{l}
                    The thesis was writen under the supervision of \\
                    \@supervisor
                \end{tabular}
            \end{flushright}
            \vspace{1cm plus 1fill}
    
            % Location and Date
            {\large Warsaw, \@date\par}
        \end{center}
        \@thanks
    \end{titlepage}%
        
    % --- Cleanup after title page ---
    \setcounter{footnote}{0}%
    \global\let\thanks\relax
    \global\let\maketitle\relax
    \global\let\@thanks\@empty
    \global\let\@author\@empty
    \global\let\@date\@empty
    \global\let\@title\@empty
    \global\let\title\relax
    \global\let\author\relax
    \global\let\date\relax
    \global\let\and\relax
}

% =========================================================================
% Custom Command Definitions for Thesis Information
% These commands should be used in the preamble of the .tex file
% =========================================================================

\def\studentID#1{\gdef\@studentID{#1}}
\def\fieldOfStudy#1{\gdef\@fieldOfStudy{#1}}
\def\supervisor#1{\gdef\@supervisor{#1}}
\def\keywords#1{\gdef\@keywords{#1}}
\def\polishTitle#1{\gdef\@polishTitle{#1}}


% =========================================================================
% Abstract Environment Redefinition
% Adds Keywords and Original Title below the abstract text on the same page.
% =========================================================================
\renewenvironment{abstract}{%
    \titlepage
    \null\nobreak\vfil
    \@beginparpenalty\@lowpenalty
    \begin{center}%
        \bfseries\large \abstractname
        \@endparpenalty\@M
    \end{center}}%
    {\par  
    \vspace*{88pt}%
    
    \begin{center}%
        \bfseries\large Keywords
        \@endparpenalty\@M
    \end{center}
    \@keywords\par
    \vspace*{88pt}%
    
    \begin{center}%
        \bfseries\large Title of the thesis in Polish language
        \@endparpenalty\@M
    \end{center}
    \@polishTitle\par
    \nobreak\vfil\null\endtitlepage\cleardoublepage}
    
    \def\cleardoublepage{\clearpage\if@twoside \ifodd\c@page\else
    \hbox{}\thispagestyle{empty}\newpage\if@twocolumn\hbox{}\newpage\fi\fi\fi}
    
\renewcommand*\@seccntformat[1]{\csname the#1\endcsname.\enspace}
\def\numberline#1{\hb@xt@\@tempdima{#1.\hfil}}
\renewcommand*\l@chapter[2]{%
  \ifnum \c@tocdepth >\m@ne
    \addpenalty{-\@highpenalty}%
    \vskip 1.0em \@plus\p@
    \setlength\@tempdima{1.5em}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      \leavevmode \bfseries
      \advance\leftskip\@tempdima
      \hskip -\leftskip
      #1\nobreak\mdseries
      \leaders\hbox{$\m@th
        \mkern \@dotsep mu\hbox{.}\mkern \@dotsep
        mu$}\hfill
      \nobreak\hb@xt@\@pnumwidth{\hss #2}\par
      \penalty\@highpenalty
    \endgroup
  \fi}

\endinput
