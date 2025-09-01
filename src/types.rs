#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeKind {
    Int,
    Char,
    Ptr,
    Array,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Type {
    pub kind: TypeKind,
    pub ptr_to: Option<Box<Type>>,
    pub array_size: Option<usize>,
}

impl Type {
    pub fn new_int() -> Self {
        Type {
            kind: TypeKind::Int,
            ptr_to: None,
            array_size: None,
        }
    }

    pub fn new_char() -> Self {
        Type {
            kind: TypeKind::Char,
            ptr_to: None,
            array_size: None,
        }
    }

    pub fn new_ptr(base_type: Type) -> Self {
        Type {
            kind: TypeKind::Ptr,
            ptr_to: Some(Box::new(base_type)),
            array_size: None,
        }
    }

    pub fn new_array(element_type: Type, size: usize) -> Self {
        Type {
            kind: TypeKind::Array,
            ptr_to: Some(Box::new(element_type)),
            array_size: Some(size),
        }
    }

    pub fn is_pointer(&self) -> bool {
        matches!(self.kind, TypeKind::Ptr)
    }

    pub fn is_array(&self) -> bool {
        matches!(self.kind, TypeKind::Array)
    }

    pub fn decay_array_to_pointer(&self) -> Type {
        match self.kind {
            TypeKind::Array => {
                if let Some(element_type) = &self.ptr_to {
                    Type::new_ptr((**element_type).clone())
                } else {
                    panic!("Invalid array type: no element type");
                }
            }
            _ => self.clone(),
        }
    }

    pub fn sizeof(&self) -> i32 {
        match self.kind {
            TypeKind::Int => 4,
            TypeKind::Char => 1,
            TypeKind::Ptr => 8,
            TypeKind::Array => {
                if let (Some(element_type), Some(array_size)) = (&self.ptr_to, self.array_size) {
                    element_type.sizeof() * array_size as i32
                } else {
                    panic!("Invalid array type");
                }
            }
        }
    }

    pub fn get_pointed_type(&self) -> &Type {
        match &self.ptr_to {
            Some(pointed_type) => pointed_type,
            None => panic!("Type is not a pointer or array"),
        }
    }
}
