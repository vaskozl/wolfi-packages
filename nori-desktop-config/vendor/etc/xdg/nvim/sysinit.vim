set number                            " Show line numbers
set relativenumber                    " Show realtive line number
set undofile                          " Persistent undo
set noswapfile                        " Disable silly .swp file
set nobackup                          " No backup file
set cursorline                        " Highlight the current line
set nofoldenable


"Set Tab to four spaces
set tabstop=4           " set hard tabstop size to 4
set softtabstop=4       " set soft tabstop size to 4
set autoindent          " automatically indent new lines
set shiftround          " when shifting, always use a multiple of shiftwidth
set expandtab           " use spaces instead of <tab>s
set smarttab            " make adding/removing tabs (spaces) smarter

set mouse=
set clipboard=unnamed

"" Make some stuff italic
highlight Comment cterm=italic gui=italic
highlight Type    cterm=italic gui=italic
highlight Keyword cterm=italic gui=italic
highlight htmlArg cterm=italic gui=italic
